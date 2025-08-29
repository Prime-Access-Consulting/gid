"""
This script processes images, generating short and long descriptions via the OpenAI API.
It can process either a single image or a folder of images.

Single image mode:
  Outputs short and long descriptions directly to standard output.

Folder mode:
  Processes images in a specified folder, storing metadata in a TSV, and optionally
  copying them into a 'Described' subfolder.

  We output four columns in the TSV:
    1) OriginalFilename
    2) ShortDescription
    3) LongDescription
    4) SHA1

Features:
  - SHA-1-based checks for previously processed images
  - Optional no-copy behavior via --no-copy
  - Control temperature and max tokens via CLI flags
  - Preserves alphabetical order of files as typically seen in Explorer/Finder
  - Shows simple progress '1 of N'
"""

import sys
import os
import re
import base64
import shutil
import hashlib
import argparse
import logging
import json
from pathlib import Path
from typing import Tuple, Set, List, Dict, Optional, Any
from dataclasses import dataclass
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Configure logging - simple output with just the message
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger('gid')

# Set OpenAI HTTP request logging to WARNING level by default (only errors shown)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Valid image extensions
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp")


class Config:
    """Class to handle configuration from config.json, environment variables, and CLI args."""
    
    DEFAULT_CONFIG = {
        "api": {
            "api_key": "",
            "model": "gpt-4o"
        },
        "parameters": {
            "temperature": 0.7,
            "max_tokens": 800
        },
        "processing": {
            "no_copy": False,
            "max_workers": None,
            "verbose": False
        },
        "output": {
            "output_folder_name": "Described",
            "tsv_filename": "descriptions.tsv"
        },
        "prompt": {
            "system_prompt": """
            You are a system generating accurate and detailed visual descriptions.
            Provided with an image, you will generate a short description of no more than 10 words and a long description which will be lengthy and detailed.
            The short description is going to be used in a filename on Windows and Mac, so no special characters or punctuation must be used that is prohibited in filenames. Never end the short description with a period or other punctuation.
            The long Description must contain as much accurate detailed information as possible. Do not start the description with "an image of" or "photo of" or anything like that, just dive into the description. The structure of a long visual description should be an overview sentence explaining the whole image followed by supporting sentences to add more detail. Always transcribe any and all text accurately and identify any famous figures or entities you are allowed to identify. Think through the description step by step and construct a well-formed description.
            Output the short description first, followed by a newline, followed by the long description.
            """,
            "short_description_max_words": 10
        }
    }
    
    @staticmethod
    def find_config_file() -> Optional[str]:
        """Find the config file in standard locations."""
        # Check current directory first
        if os.path.exists("config.json"):
            return "config.json"
        
        # Check user config directory
        user_config_dir = os.path.expanduser("~/.config/gid")
        user_config_file = os.path.join(user_config_dir, "config.json")
        if os.path.exists(user_config_file):
            return user_config_file
        
        return None
    
    @staticmethod
    def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from a JSON file."""
        config = Config.DEFAULT_CONFIG.copy()
        
        # If no config path provided, try to find one
        if not config_path:
            config_path = Config.find_config_file()
        
        # If we found a config file, load and merge it
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                
                # Deep merge the user config into the default config
                Config._merge_configs(config, user_config)
                
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.warning(f"Error loading config from {config_path}: {str(e)}")
        
        return config
    
    @staticmethod
    def _merge_configs(target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Recursively merge source config into target config."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                Config._merge_configs(target[key], value)
            else:
                target[key] = value


@dataclass
class ImageResult:
    """Data class to store image processing results."""
    idx: int
    original_filename: str
    image_path: str
    file_hash: str
    short_desc: str
    long_desc: str


class FileHelper:
    """Helper class for file operations."""
    
    @staticmethod
    def hash_file(file_path: str, chunk_size: int = 65536) -> str:
        """
        Compute SHA-1 hash of a file to facilitate binary comparison.
        """
        sha1 = hashlib.sha1()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                sha1.update(chunk)
        return sha1.hexdigest()
    
    @staticmethod
    def encode_image(image_path: str) -> str:
        """
        Encode the given image file into base64.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    @staticmethod
    def sanitize_filename(name: str) -> str:
        """
        Replace invalid Windows filename characters (\\/:*?"<>|) with underscores,
        and remove trailing periods/spaces.
        """
        name = re.sub(r'[\\/:*?"<>|]', "_", name)
        return name.rstrip(" .")
    
    @staticmethod
    def ensure_described_folder(folder_path: str, output_folder_name: str, no_copy: bool = False) -> str:
        """
        Create output subfolder if it does not exist, unless no_copy is True.
        
        Args:
            folder_path: Source folder path
            output_folder_name: Name of output subfolder
            no_copy: If True, return source folder directly (no subfolder created)
        
        Returns:
            Path where TSV and copied files should be placed
        """
        if no_copy:
            # When no_copy is True, use the source folder directly
            return folder_path
        else:
            # Original behavior: create and return subfolder path
            described_folder_path = os.path.join(folder_path, output_folder_name)
            if not os.path.isdir(described_folder_path):
                os.mkdir(described_folder_path)
            return described_folder_path


class TSVHandler:
    """Class to handle TSV file operations."""
    
    def __init__(self, tsv_path: str):
        self.tsv_path = tsv_path
        self.existing_lines: Set[str] = set()
        self.known_hashes: Set[str] = set()
        self.load()
    
    @staticmethod
    def _escape_newlines(text: str) -> str:
        """Escape actual newlines as \\n for TSV storage"""
        return text.replace('\n', '\\n').replace('\r', '\\r')
    
    @staticmethod
    def _unescape_newlines(text: str) -> str:
        """Unescape \\n back to actual newlines when reading"""
        return text.replace('\\n', '\n').replace('\\r', '\r')
    
    def load(self) -> None:
        """
        Load existing lines (4 columns) from the TSV (skipping header).
        existing_lines -> set of raw lines
        known_hashes -> set of SHA-1 strings from the 4th column
        """
        if not os.path.exists(self.tsv_path):
            return
        
        with open(self.tsv_path, "r", encoding="utf-8") as tsv_file:
            # Skip header
            next(tsv_file, None)
            for line in tsv_file:
                line = line.strip()
                if not line:
                    continue
                self.existing_lines.add(line)
                parts = line.split("\t")
                if len(parts) == 4:
                    hash_value = parts[3]
                    self.known_hashes.add(hash_value)
    
    def write_header(self) -> None:
        """Write TSV header if the file is new."""
        is_new_tsv = not os.path.exists(self.tsv_path)
        if is_new_tsv:
            with open(self.tsv_path, "w", encoding="utf-8") as tsv_file:
                tsv_file.write("OriginalFilename\tShortDescription\tLongDescription\tSHA1\n")
    
    def add_entry(self, orig_filename: str, short_desc: str, long_desc: str, file_hash: str) -> None:
        """Add an entry to the TSV file."""
        # Escape newlines in descriptions for proper TSV format
        escaped_short = self._escape_newlines(short_desc)
        escaped_long = self._escape_newlines(long_desc)
        line_to_write = f"{orig_filename}\t{escaped_short}\t{escaped_long}\t{file_hash}"
        
        if line_to_write not in self.existing_lines:
            with open(self.tsv_path, "a", encoding="utf-8") as tsv_file:
                tsv_file.write(line_to_write + "\n")
            self.existing_lines.add(line_to_write)
            self.known_hashes.add(file_hash)


class ImageDescriber:
    """Class to interact with OpenAI API for image description."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o", temperature: float = 0.7, max_tokens: int = 800, system_prompt: Optional[str] = None):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key)
        self.system_prompt = system_prompt or Config.DEFAULT_CONFIG["prompt"]["system_prompt"]
    
    def describe_image(self, image_path: str) -> Tuple[str, str]:
        """
        Call the OpenAI API to generate short and long descriptions of an image.
        Returns (short_desc, long_desc).
        """
        try:
            encoded_image = FileHelper.encode_image(image_path)
            extension = os.path.splitext(image_path)[1][1:].lower()
            data_url = f"data:image/{extension};base64,{encoded_image}"

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe the following image."},
                            {"type": "image_url", "image_url": {"url": data_url}}
                        ]
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            text_response = response.choices[0].message.content.strip()
            short_desc, long_desc = text_response.split("\n", 1)
            return short_desc.strip(), long_desc.strip()
        except Exception as e:
            logger.error(f"Error describing image {image_path}: {str(e)}")
            return "Error", f"Error: {str(e)}"


class ImageProcessor:
    """Main class to orchestrate image processing."""
    
    def __init__(self, folder_path: str, config: Dict[str, Any]):
        self.folder_path = folder_path
        self.api_key = config["api"]["api_key"]
        self.model = config["api"]["model"]
        self.temperature = config["parameters"]["temperature"]
        self.max_tokens = config["parameters"]["max_tokens"]
        self.no_copy = config["processing"]["no_copy"]
        self.max_workers = config["processing"]["max_workers"]
        self.verbose = config["processing"]["verbose"]
        self.output_folder_name = config["output"]["output_folder_name"]
        self.tsv_filename = config["output"]["tsv_filename"]
        self.system_prompt = config["prompt"]["system_prompt"]
        
        self.described_folder_path = FileHelper.ensure_described_folder(folder_path, self.output_folder_name, self.no_copy)
        self.tsv_path = os.path.join(self.described_folder_path, self.tsv_filename)
        self.tsv_handler = TSVHandler(self.tsv_path)
        self.describer = ImageDescriber(
            api_key=self.api_key,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            system_prompt=self.system_prompt
        )
        self.progress_lock = Lock()
        
        # Set up logging based on verbosity
        if self.verbose:
            # Enable HTTP request logging in verbose mode
            logging.getLogger("openai").setLevel(logging.INFO)
            logging.getLogger("httpx").setLevel(logging.INFO)
    
    def collect_image_files(self) -> List[Tuple[int, str, str, str]]:
        """
        Collect image files that need processing.
        Returns a list of tuples: (idx, filename, image_path, file_hash)
        """
        # Sort by name ignoring case, to mimic typical Explorer/Finder order
        all_files = sorted(os.listdir(self.folder_path), key=str.lower)
        
        tasks = []
        pending_hashes = set()
        
        # Identify which files need describing, preserving sorted order
        for idx, filename in enumerate(all_files):
            if filename.lower().endswith(VALID_EXTENSIONS):
                image_path = os.path.join(self.folder_path, filename)
                file_hash = FileHelper.hash_file(image_path)
                # Skip if known or queued
                if file_hash in self.tsv_handler.known_hashes or file_hash in pending_hashes:
                    continue
                pending_hashes.add(file_hash)
                tasks.append((idx, filename, image_path, file_hash))
        
        return tasks
    
    def process_image(self, task: Tuple[int, str, str, str]) -> Optional[ImageResult]:
        """Process a single image and return the result."""
        idx, filename, image_path, file_hash = task
        short_desc, long_desc = self.describer.describe_image(image_path)
        
        if short_desc == "Error":
            return None
        
        return ImageResult(
            idx=idx,
            original_filename=filename,
            image_path=image_path,
            file_hash=file_hash,
            short_desc=short_desc,
            long_desc=long_desc
        )
    
    def handle_image_result(self, result: ImageResult) -> None:
        """
        Handle the image processing result - copy file and add to TSV.
        """
        # Sanitize short_desc for file naming
        short_desc = FileHelper.sanitize_filename(result.short_desc)
        
        # Build new filename
        _, ext = os.path.splitext(result.image_path)
        base_name = short_desc
        new_image_name = f"{base_name}{ext.lower()}"
        new_image_path = os.path.join(self.described_folder_path, new_image_name)
        
        # Handle collisions - limit to 100 attempts to avoid infinite loop
        counter = 2
        max_attempts = 100
        while os.path.exists(new_image_path) and counter <= max_attempts:
            alt_name = f"{base_name} {counter}{ext.lower()}"
            alt_path = os.path.join(self.described_folder_path, alt_name)
            if not os.path.exists(alt_path):
                new_image_name = alt_name
                new_image_path = alt_path
                break
            counter += 1
        
        # If we couldn't find a non-colliding name, skip this file
        if counter > max_attempts:
            logger.warning(f"Could not find unique name for {result.image_path} after {max_attempts} attempts. Skipping.")
            return
        
        # Copy file unless no_copy is set
        if not self.no_copy:
            try:
                shutil.copy2(result.image_path, new_image_path)
            except Exception as e:
                logger.error(f"Error copying {result.image_path} to {new_image_path}: {str(e)}")
                return
        
        # Final short name is the new image's base
        final_short_name = os.path.splitext(new_image_name)[0]
        
        # Add to TSV
        self.tsv_handler.add_entry(
            result.original_filename,
            final_short_name,
            result.long_desc,
            result.file_hash
        )
    
    def process_all(self) -> None:
        """Process all images in the folder."""
        tasks = self.collect_image_files()
        total_count = len(tasks)
        
        if total_count == 0:
            logger.info("No new images to process.")
            return
        
        logger.info(f"Processing {total_count} images...")
        
        # Ensure TSV header exists
        self.tsv_handler.write_header()
        
        # We'll use this to track progress
        done_count = 0
        
        # Parallel describing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {executor.submit(self.process_image, task): task for task in tasks}
            
            for future in as_completed(future_to_task):
                result = future.result()
                
                with self.progress_lock:
                    done_count += 1
                    logger.info(f"{done_count} of {total_count}")
                
                if result:
                    self.handle_image_result(result)
        
        logger.info(f"Processed {done_count} images.")


class CLI:
    """Command-line interface handler."""
    
    @staticmethod
    def parse_args() -> argparse.Namespace:
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser(
            description="Describe images with OpenAI. For folders, writes 4 columns in TSV:\n"
                        "OriginalFilename, ShortDescription, LongDescription, SHA1.\n"
                        "For single image files, outputs descriptions directly."
        )
        parser.add_argument("path", nargs="?", help="Path to folder or single image file.")
        parser.add_argument(
            "-t", "--temperature",
            type=float,
            help="Sampling temperature for OpenAI (default=0.7)."
        )
        parser.add_argument(
            "-l", "--length",
            type=int,
            help="Max tokens (default=800)."
        )
        parser.add_argument(
            "-n", "--no-copy",
            action="store_true",
            help="If provided, do NOT copy files to the output folder (folder mode only)."
        )
        parser.add_argument(
            "-k", "--api-key",
            type=str,
            help="OpenAI API key (overrides config file and environment variable)."
        )
        parser.add_argument(
            "-w", "--workers",
            type=int,
            help="Maximum number of concurrent workers (folder mode only)."
        )
        parser.add_argument(
            "-v", "--verbose",
            action="store_true",
            help="Enable verbose output including HTTP requests."
        )
        parser.add_argument(
            "-c", "--config",
            type=str,
            help="Path to the configuration file (default: config.json in the current directory or ~/.config/gid/config.json)"
        )
        parser.add_argument(
            "-m", "--model",
            type=str,
            help="OpenAI model to use (default: gpt-4o)"
        )
        
        # If user didn't supply any arguments, show help and exit
        if len(sys.argv) == 1:
            parser.print_help(sys.stderr)
            sys.exit(1)
        
        return parser.parse_args()
    
    @staticmethod
    def get_config(args: argparse.Namespace) -> Dict[str, Any]:
        """Load config from file and override with command line args."""
        # Load config from file
        config = Config.load_config(args.config)
        
        # Override with command line args
        if args.api_key:
            config["api"]["api_key"] = args.api_key
        if args.model:
            config["api"]["model"] = args.model
        if args.temperature is not None:
            config["parameters"]["temperature"] = args.temperature
        if args.length is not None:
            config["parameters"]["max_tokens"] = args.length
        if args.no_copy:
            config["processing"]["no_copy"] = True
        if args.workers is not None:
            config["processing"]["max_workers"] = args.workers
        if args.verbose:
            config["processing"]["verbose"] = True
        
        # Check for API key in environment if not in config or args
        if not config["api"]["api_key"]:
            config["api"]["api_key"] = os.environ.get("OPENAI_API_KEY", "")
        
        return config
    
    @staticmethod
    def process_single_image(image_path: str, config: Dict[str, Any]) -> None:
        """Process a single image file and output descriptions directly."""
        if config["processing"]["verbose"]:
            # Enable HTTP request logging in verbose mode
            logging.getLogger("openai").setLevel(logging.INFO)
            logging.getLogger("httpx").setLevel(logging.INFO)
            
        describer = ImageDescriber(
            api_key=config["api"]["api_key"],
            model=config["api"]["model"],
            temperature=config["parameters"]["temperature"],
            max_tokens=config["parameters"]["max_tokens"],
            system_prompt=config["prompt"]["system_prompt"]
        )
        
        short_desc, long_desc = describer.describe_image(image_path)
        
        if short_desc == "Error":
            print(f"Error: {long_desc}")
            sys.exit(1)
            
        print(short_desc)
        print(long_desc)
    
    @staticmethod
    def run() -> None:
        """Run the command-line interface."""
        args = CLI.parse_args()
        
        # If user didn't supply the path argument, show usage and exit
        if not args.path:
            print("Error: Path to folder or image file is required.", file=sys.stderr)
            sys.exit(1)
        
        # Get configuration
        config = CLI.get_config(args)
        
        # Check for API key
        if not config["api"]["api_key"]:
            print("Error: OpenAI API key not provided. Use -k/--api-key, set OPENAI_API_KEY environment variable, or add it to config.json.", file=sys.stderr)
            sys.exit(1)
        
        # Check if path is a directory or a file
        if os.path.isdir(args.path):
            # Process folder
            processor = ImageProcessor(
                folder_path=args.path,
                config=config
            )
            processor.process_all()
        elif os.path.isfile(args.path) and args.path.lower().endswith(VALID_EXTENSIONS):
            # Process single image
            CLI.process_single_image(
                image_path=args.path,
                config=config
            )
        else:
            print(f"Error: '{args.path}' is not a valid directory or image file.", file=sys.stderr)
            sys.exit(1)


def main():
    """Main entry point."""
    CLI.run()


if __name__ == "__main__":
    main()