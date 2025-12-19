"""
This script processes images, generating short and long descriptions via the OpenAI API.
It can process either a single image or a folder of images.

Single image mode:
  Outputs short and long descriptions directly to standard output.

Folder mode:
  Processes images in a specified folder, storing metadata in a TSV, and optionally
  copying them into a 'Described' subfolder.

  We output five columns in the TSV:
    1) OriginalFilename
    2) ShortDescription
    3) LongDescription
    4) Context
    5) SHA1

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
from typing import Tuple, List, Dict, Optional, Any
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
            "model": "gpt-5.2"
        },
        "parameters": {
            "temperature": 0.7,
            "max_tokens": 4000
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
            The long description must contain as much accurate detailed information as possible. Do not start the description with "an image of" or "photo of" or anything like that, just dive into the description. The structure of a long visual description should be an overview sentence explaining the whole image followed by supporting sentences to add more detail. Always transcribe any and all text accurately and identify any famous figures or entities you are allowed to identify. Think through the description step by step and construct a well-formed description.
            Output exactly two lines in this format:
            SHORT: <short description>
            LONG: <long description>
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
    original_filename: str
    image_path: str
    file_hash: str
    short_desc: str
    long_desc: str
    context: str


@dataclass
class TSVEntry:
    """Data class to store TSV row data."""
    original_filename: str
    short_desc: str
    long_desc: str
    context: str
    file_hash: str


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
        self.entries_by_hash: Dict[str, TSVEntry] = {}
        self.hash_order: List[str] = []
        self.has_context_column = False
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
        Load existing lines from the TSV (skipping header).
        """
        if not os.path.exists(self.tsv_path):
            return
        
        with open(self.tsv_path, "r", encoding="utf-8") as tsv_file:
            header = next(tsv_file, None)
            if header:
                header_cols = header.strip().split("\t")
                self.has_context_column = "Context" in header_cols
            for line in tsv_file:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if self.has_context_column and len(parts) >= 5:
                    orig_filename, short_desc, long_desc, context, hash_value = parts[:5]
                elif len(parts) >= 4:
                    orig_filename, short_desc, long_desc, hash_value = parts[:4]
                    context = ""
                else:
                    continue
                entry = TSVEntry(
                    original_filename=orig_filename,
                    short_desc=self._unescape_newlines(short_desc),
                    long_desc=self._unescape_newlines(long_desc),
                    context=self._unescape_newlines(context),
                    file_hash=hash_value
                )
                if hash_value not in self.entries_by_hash:
                    self.entries_by_hash[hash_value] = entry
                    self.hash_order.append(hash_value)

    def get_entry(self, file_hash: str) -> Optional[TSVEntry]:
        """Get an entry by hash if it exists."""
        return self.entries_by_hash.get(file_hash)

    def upsert_entry(
        self,
        orig_filename: str,
        short_desc: str,
        long_desc: str,
        context: str,
        file_hash: str
    ) -> None:
        """Insert or update an entry keyed by file hash."""
        entry = self.entries_by_hash.get(file_hash)
        if entry is None:
            self.entries_by_hash[file_hash] = TSVEntry(
                original_filename=orig_filename,
                short_desc=short_desc,
                long_desc=long_desc,
                context=context,
                file_hash=file_hash
            )
            self.hash_order.append(file_hash)
            return

        if not entry.original_filename:
            entry.original_filename = orig_filename
        if short_desc:
            entry.short_desc = short_desc
        if long_desc:
            entry.long_desc = long_desc
        if context:
            entry.context = context

    def write_all(self) -> None:
        """Rewrite the TSV with the current entries."""
        with open(self.tsv_path, "w", encoding="utf-8") as tsv_file:
            tsv_file.write("OriginalFilename\tShortDescription\tLongDescription\tContext\tSHA1\n")
            for hash_value in self.hash_order:
                entry = self.entries_by_hash.get(hash_value)
                if not entry:
                    continue
                escaped_short = self._escape_newlines(entry.short_desc)
                escaped_long = self._escape_newlines(entry.long_desc)
                escaped_context = self._escape_newlines(entry.context)
                line = (
                    f"{entry.original_filename}\t"
                    f"{escaped_short}\t"
                    f"{escaped_long}\t"
                    f"{escaped_context}\t"
                    f"{entry.file_hash}"
                )
                tsv_file.write(line + "\n")


class ImageDescriber:
    """Class to interact with OpenAI API for image description."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5.2",
        temperature: float = 0.7,
        max_tokens: int = 4000,
        system_prompt: Optional[str] = None,
        short_description_max_words: Optional[int] = None
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key)
        self.system_prompt = system_prompt or Config.DEFAULT_CONFIG["prompt"]["system_prompt"]
        self.short_description_max_words = short_description_max_words
    
    def _limit_short_description(self, short_desc: str) -> str:
        """Trim short description to the configured max word count."""
        max_words = self.short_description_max_words
        if not max_words or max_words <= 0:
            return short_desc
        words = short_desc.split()
        if len(words) <= max_words:
            return short_desc
        return " ".join(words[:max_words])
    
    def describe_image(self, image_path: str, context: str = "") -> Tuple[str, str]:
        """
        Call the OpenAI API to generate short and long descriptions of an image.
        Returns (short_desc, long_desc).
        """
        try:
            encoded_image = FileHelper.encode_image(image_path)
            extension = os.path.splitext(image_path)[1][1:].lower()
            data_url = f"data:image/{extension};base64,{encoded_image}"

            instructions = self.system_prompt
            if context.strip():
                instructions = (
                    f"{instructions.strip()}\n\n"
                    "Additional image facts provided by the user (treat as true): "
                    f"{context.strip()}"
                )

            response = self.client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Describe the following image."},
                            {"type": "input_image", "image_url": data_url}
                        ]
                    }
                ],
                instructions=instructions,
                max_output_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            text_response = response.output_text
            if text_response is None:
                logger.error(f"Empty response from API for {image_path}")
                return "Error", "Empty response from API"
            text_response = text_response.strip()

            # Parse response format: "SHORT: ... LONG: ..."
            match = re.search(
                r"(?is)short\\s*:\\s*(.*?)\\s*long\\s*:\\s*(.*)",
                text_response
            )
            if match:
                short_part = match.group(1).strip()
                long_part = match.group(2).strip()
                short_part = self._limit_short_description(short_part)
                return short_part, long_part

            # Parse two-line format: first non-empty line is short, rest is long
            lines = [line.strip() for line in text_response.splitlines() if line.strip()]
            if len(lines) >= 2:
                short_part = re.sub(r"^short( description)?\\s*:\\s*", "", lines[0], flags=re.IGNORECASE).strip()
                long_part = "\n".join(lines[1:]).strip()
                long_part = re.sub(r"^long( description)?\\s*:\\s*", "", long_part, flags=re.IGNORECASE).strip()
                short_part = self._limit_short_description(short_part)
                if long_part:
                    return short_part, long_part

            # Fallback if format is unexpected
            logger.warning(f"Unexpected response format for {image_path}")
            words = text_response.split()
            max_words = self.short_description_max_words or 10
            short_desc = " ".join(words[:max_words])
            short_desc = self._limit_short_description(short_desc)
            return short_desc, text_response
        except Exception as e:
            logger.error(f"Error describing image {image_path}: {str(e)}")
            return "Error", f"Error: {str(e)}"


class ImageProcessor:
    """Main class to orchestrate image processing."""
    
    def __init__(self, folder_path: str, config: Dict[str, Any], init_only: bool = False):
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
        self.short_description_max_words = config["prompt"].get("short_description_max_words")
        
        self.described_folder_path = FileHelper.ensure_described_folder(folder_path, self.output_folder_name, self.no_copy)
        self.tsv_path = os.path.join(self.described_folder_path, self.tsv_filename)
        self.tsv_handler = TSVHandler(self.tsv_path)
        self.describer = None
        if not init_only:
            self.describer = ImageDescriber(
                api_key=self.api_key,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                system_prompt=self.system_prompt,
                short_description_max_words=self.short_description_max_words
            )
        self.progress_lock = Lock()
        
        # Set up logging based on verbosity
        if self.verbose:
            # Enable HTTP request logging in verbose mode
            logging.getLogger("openai").setLevel(logging.INFO)
            logging.getLogger("httpx").setLevel(logging.INFO)
    
    def collect_image_files(self, include_existing: bool = False) -> List[Tuple[str, str, str, str]]:
        """
        Collect image files that need processing.
        Returns a list of tuples: (filename, image_path, file_hash, context)
        """
        # Sort by name ignoring case, to mimic typical Explorer/Finder order
        all_files = sorted(os.listdir(self.folder_path), key=str.lower)
        
        tasks = []
        pending_hashes = set()
        
        # Identify which files need describing, preserving sorted order
        for filename in all_files:
            if filename.lower().endswith(VALID_EXTENSIONS):
                image_path = os.path.join(self.folder_path, filename)
                file_hash = FileHelper.hash_file(image_path)
                entry = self.tsv_handler.get_entry(file_hash)
                if entry and entry.short_desc and entry.long_desc and not include_existing:
                    continue
                # Skip if queued
                if file_hash in pending_hashes:
                    continue
                pending_hashes.add(file_hash)
                context = entry.context if entry else ""
                tasks.append((filename, image_path, file_hash, context))
        
        return tasks
    
    def process_image(self, task: Tuple[str, str, str, str]) -> Optional[ImageResult]:
        """Process a single image and return the result."""
        filename, image_path, file_hash, context = task
        if not self.describer:
            logger.error("Image describer is not initialized.")
            return None
        short_desc, long_desc = self.describer.describe_image(image_path, context)
        
        if short_desc == "Error":
            return None
        
        return ImageResult(
            original_filename=filename,
            image_path=image_path,
            file_hash=file_hash,
            short_desc=short_desc,
            long_desc=long_desc,
            context=context
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
        self.tsv_handler.upsert_entry(
            result.original_filename,
            final_short_name,
            result.long_desc,
            result.context,
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

        # Rewrite TSV with updated entries
        self.tsv_handler.write_all()
        logger.info(f"Processed {done_count} images.")

    def init_tsv(self) -> None:
        """Initialize a TSV with hashes and empty descriptions/context."""
        all_files = sorted(os.listdir(self.folder_path), key=str.lower)
        pending_hashes = set()
        self.tsv_handler.entries_by_hash.clear()
        self.tsv_handler.hash_order.clear()

        total_count = 0
        for filename in all_files:
            if not filename.lower().endswith(VALID_EXTENSIONS):
                continue
            image_path = os.path.join(self.folder_path, filename)
            file_hash = FileHelper.hash_file(image_path)
            if file_hash in pending_hashes:
                continue
            pending_hashes.add(file_hash)
            total_count += 1
            self.tsv_handler.upsert_entry(
                orig_filename=filename,
                short_desc="",
                long_desc="",
                context="",
                file_hash=file_hash
            )

        if total_count == 0:
            logger.info("No images found to initialize.")
            return
        self.tsv_handler.write_all()
        logger.info(f"Initialized TSV for {total_count} images.")


class CLI:
    """Command-line interface handler."""
    
    @staticmethod
    def parse_args() -> argparse.Namespace:
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser(
            description="Describe images with OpenAI. For folders, writes 5 columns in TSV:\n"
                        "OriginalFilename, ShortDescription, LongDescription, Context, SHA1.\n"
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
            help="Max tokens (default=4000)."
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
            "--init-tsv",
            action="store_true",
            help="Generate TSV with hashes and empty descriptions/context (folder mode only)."
        )
        parser.add_argument(
            "-m", "--model",
            type=str,
            help="OpenAI model to use (default: gpt-5.2)"
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
            system_prompt=config["prompt"]["system_prompt"],
            short_description_max_words=config["prompt"].get("short_description_max_words")
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
        
        # Check for API key unless we're initializing a TSV
        if not args.init_tsv and not config["api"]["api_key"]:
            print("Error: OpenAI API key not provided. Use -k/--api-key, set OPENAI_API_KEY environment variable, or add it to config.json.", file=sys.stderr)
            sys.exit(1)
        
        # Check if path is a directory or a file
        if os.path.isdir(args.path):
            if args.init_tsv:
                processor = ImageProcessor(
                    folder_path=args.path,
                    config=config,
                    init_only=True
                )
                processor.init_tsv()
            else:
                # Process folder
                processor = ImageProcessor(
                    folder_path=args.path,
                    config=config
                )
                processor.process_all()
        elif os.path.isfile(args.path) and args.path.lower().endswith(VALID_EXTENSIONS):
            if args.init_tsv:
                print("Error: --init-tsv is only supported for folder mode.", file=sys.stderr)
                sys.exit(1)
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
