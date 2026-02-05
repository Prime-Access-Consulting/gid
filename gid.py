"""
This script processes images, generating short and long descriptions via the OpenAI API.
It can process either a single image or a folder of images.

Single image mode:
  Outputs short and long descriptions directly to standard output.

Folder mode:
  Processes images in a specified folder, storing metadata in a TSV, and optionally
  copying them into a 'Described' subfolder.

  We output six columns in the TSV:
    1) OriginalFilename
    2) ShortDescription
    3) LongDescription
    4) Context
    5) Composite
    6) SHA1

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
from typing import Tuple, List, Dict, Optional, Any, Set
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
            "temperature": 1.0,
            "max_tokens": 4000
        },
        "processing": {
            "no_copy": False,
            "no_composites": False,
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
            "single_image_prompt": "Describe the following image.",
            "composite_image_prompt": "Describe the following images together as a single composite.",
            "context_template": "Additional image facts provided by the user (treat as true): {context}",
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
    composite: bool
    file_hash: str


@dataclass
class CompositeResult:
    """Data class to store composite image processing results."""
    entry: TSVEntry
    composite_hash: str
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
        self.entries: List[TSVEntry] = []
        self.entries_by_hash: Dict[str, TSVEntry] = {}
        self.has_context_column = False
        self.has_composite_column = False
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
            header_cols = header.strip().split("\t") if header else []
            col_index = {col: idx for idx, col in enumerate(header_cols)}
            if header_cols:
                self.has_context_column = "Context" in col_index
                self.has_composite_column = "Composite" in col_index
            for line in tsv_file:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if not header_cols:
                    if len(parts) >= 6:
                        orig_filename, short_desc, long_desc, context, composite_raw, hash_value = parts[:6]
                    elif len(parts) >= 5:
                        orig_filename, short_desc, long_desc, context, hash_value = parts[:5]
                        composite_raw = ""
                    elif len(parts) >= 4:
                        orig_filename, short_desc, long_desc, hash_value = parts[:4]
                        context = ""
                        composite_raw = ""
                    else:
                        continue
                else:
                    def col(name: str, default: str = "") -> str:
                        idx = col_index.get(name)
                        if idx is None or idx >= len(parts):
                            return default
                        return parts[idx]

                    orig_filename = col("OriginalFilename", parts[0] if parts else "")
                    short_desc = col("ShortDescription", parts[1] if len(parts) > 1 else "")
                    long_desc = col("LongDescription", parts[2] if len(parts) > 2 else "")
                    context = col("Context", "")
                    composite_raw = col("Composite", "")
                    hash_value = col("SHA1", parts[-1] if parts else "")

                composite = composite_raw.strip().lower() == "yes"
                entry = TSVEntry(
                    original_filename=orig_filename,
                    short_desc=self._unescape_newlines(short_desc),
                    long_desc=self._unescape_newlines(long_desc),
                    context=self._unescape_newlines(context),
                    composite=composite,
                    file_hash=hash_value
                )
                self.entries.append(entry)
                if hash_value and hash_value not in self.entries_by_hash:
                    self.entries_by_hash[hash_value] = entry

    def get_entry(self, file_hash: str) -> Optional[TSVEntry]:
        """Get an entry by hash if it exists."""
        return self.entries_by_hash.get(file_hash)

    def get_composite_entries(self) -> List[TSVEntry]:
        """Return all composite entries."""
        return [entry for entry in self.entries if entry.composite]

    def update_entry_hash(self, entry: TSVEntry, new_hash: str) -> None:
        """Update the hash for an entry and maintain the lookup map."""
        old_hash = entry.file_hash
        if old_hash and self.entries_by_hash.get(old_hash) is entry:
            del self.entries_by_hash[old_hash]
        entry.file_hash = new_hash
        if new_hash and new_hash not in self.entries_by_hash:
            self.entries_by_hash[new_hash] = entry

    def upsert_entry(
        self,
        orig_filename: str,
        short_desc: str,
        long_desc: str,
        context: str,
        file_hash: str,
        composite: bool = False
    ) -> None:
        """Insert or update an entry keyed by file hash."""
        entry = self.entries_by_hash.get(file_hash)
        if entry is None:
            entry = TSVEntry(
                original_filename=orig_filename,
                short_desc=short_desc,
                long_desc=long_desc,
                context=context,
                composite=composite,
                file_hash=file_hash
            )
            self.entries.append(entry)
            if file_hash:
                self.entries_by_hash[file_hash] = entry
            return

        if not entry.original_filename:
            entry.original_filename = orig_filename
        if short_desc:
            entry.short_desc = short_desc
        if long_desc:
            entry.long_desc = long_desc
        if context:
            entry.context = context
        entry.composite = entry.composite or composite
        if file_hash and file_hash not in self.entries_by_hash:
            self.entries_by_hash[file_hash] = entry

    def write_all(self) -> None:
        """Rewrite the TSV with the current entries."""
        with open(self.tsv_path, "w", encoding="utf-8") as tsv_file:
            tsv_file.write("OriginalFilename\tShortDescription\tLongDescription\tContext\tComposite\tSHA1\n")
            for entry in self.entries:
                escaped_short = self._escape_newlines(entry.short_desc)
                escaped_long = self._escape_newlines(entry.long_desc)
                escaped_context = self._escape_newlines(entry.context)
                composite_value = "yes" if entry.composite else "no"
                line = (
                    f"{entry.original_filename}\t"
                    f"{escaped_short}\t"
                    f"{escaped_long}\t"
                    f"{escaped_context}\t"
                    f"{composite_value}\t"
                    f"{entry.file_hash}"
                )
                tsv_file.write(line + "\n")


class ImageDescriber:
    """Class to interact with OpenAI API for image description."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5.2",
        temperature: float = 1.0,
        max_tokens: int = 4000,
        system_prompt: Optional[str] = None,
        single_image_prompt: Optional[str] = None,
        composite_image_prompt: Optional[str] = None,
        context_template: Optional[str] = None,
        short_description_max_words: Optional[int] = None
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key)
        default_prompt = Config.DEFAULT_CONFIG["prompt"]
        self.system_prompt = system_prompt if system_prompt is not None else default_prompt["system_prompt"]
        self.single_image_prompt = (
            single_image_prompt
            if single_image_prompt is not None
            else default_prompt["single_image_prompt"]
        )
        self.composite_image_prompt = (
            composite_image_prompt
            if composite_image_prompt is not None
            else default_prompt["composite_image_prompt"]
        )
        self.context_template = (
            context_template
            if context_template is not None
            else default_prompt["context_template"]
        )
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
        """Call the OpenAI API to describe a single image."""
        return self.describe_images([image_path], context)

    def describe_images(self, image_paths: List[str], context: str = "") -> Tuple[str, str]:
        """
        Call the OpenAI API to generate short and long descriptions of one or more images.
        Returns (short_desc, long_desc).
        """
        try:
            content = []
            prompt_text = self.single_image_prompt
            if len(image_paths) > 1:
                prompt_text = self.composite_image_prompt
            content.append({"type": "input_text", "text": prompt_text})
            for image_path in image_paths:
                encoded_image = FileHelper.encode_image(image_path)
                extension = os.path.splitext(image_path)[1][1:].lower()
                data_url = f"data:image/{extension};base64,{encoded_image}"
                content.append({"type": "input_image", "image_url": data_url})

            instructions = self.system_prompt
            if context.strip():
                context_value = context.strip()
                template = self.context_template or ""
                if "{context}" in template:
                    context_text = template.replace("{context}", context_value)
                else:
                    context_text = f"{template} {context_value}".strip() if template else context_value
                instructions = f"{instructions.strip()}\n\n{context_text}"

            response_params = {
                "model": self.model,
                "input": [
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                "instructions": instructions,
                "max_output_tokens": self.max_tokens
            }
            if self.temperature is not None and self.temperature != 1.0:
                response_params["temperature"] = self.temperature

            response = self.client.responses.create(**response_params)
            
            text_response = response.output_text
            if text_response is None:
                logger.error("Empty response from API.")
                return "Error", "Empty response from API"
            text_response = text_response.strip()

            label_pattern = r"\b{label}\b(?:\s+description)?\s*[:\-\u2013\u2014\uFF1A]?"
            short_match = re.search(label_pattern.format(label="short"), text_response, flags=re.IGNORECASE)
            long_match = re.search(label_pattern.format(label="long"), text_response, flags=re.IGNORECASE)
            if short_match and long_match and long_match.start() > short_match.end():
                short_part = text_response[short_match.end():long_match.start()].strip()
                long_part = text_response[long_match.end():].strip()
                if short_part and long_part:
                    short_part = self._limit_short_description(short_part)
                    return short_part, long_part

            # Parse two-line format: first non-empty line is short, rest is long
            lines = [line.strip() for line in text_response.splitlines() if line.strip()]
            if len(lines) >= 2:
                short_part = re.sub(
                    label_pattern.format(label="short"),
                    "",
                    lines[0],
                    flags=re.IGNORECASE
                ).strip()
                long_part = "\n".join(lines[1:]).strip()
                long_part = re.sub(
                    label_pattern.format(label="long"),
                    "",
                    long_part,
                    flags=re.IGNORECASE
                ).strip()
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
        self.no_composites = config["processing"].get("no_composites", False)
        self.max_workers = config["processing"]["max_workers"]
        self.verbose = config["processing"]["verbose"]
        self.output_folder_name = config["output"]["output_folder_name"]
        self.tsv_filename = config["output"]["tsv_filename"]
        self.system_prompt = config["prompt"]["system_prompt"]
        self.single_image_prompt = config["prompt"].get("single_image_prompt")
        self.composite_image_prompt = config["prompt"].get("composite_image_prompt")
        self.context_template = config["prompt"].get("context_template")
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
                single_image_prompt=self.single_image_prompt,
                composite_image_prompt=self.composite_image_prompt,
                context_template=self.context_template,
                short_description_max_words=self.short_description_max_words
            )
        self.progress_lock = Lock()
        
        # Set up logging based on verbosity
        if self.verbose:
            # Enable HTTP request logging in verbose mode
            logging.getLogger("openai").setLevel(logging.INFO)
            logging.getLogger("httpx").setLevel(logging.INFO)
    
    def collect_image_files(
        self,
        include_existing: bool = False,
        skip_filenames: Optional[Set[str]] = None
    ) -> List[Tuple[str, str, str, str]]:
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
                if skip_filenames and filename.lower() in skip_filenames:
                    continue
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

    @staticmethod
    def _normalize_base_name(name: str) -> str:
        """Normalize a composite base name for matching."""
        return Path(name).stem.lower()

    def _discover_composite_sets(self) -> List[Tuple[str, List[Tuple[int, str, str, str]]]]:
        """Discover composite sets based on underscore-numbered filenames."""
        pattern = re.compile(
            r"^(?P<base>.+)_(?P<index>\d+)\.([A-Za-z0-9]+)$",
            re.IGNORECASE
        )
        grouped: Dict[str, Dict[str, Any]] = {}
        all_filenames = [
            filename
            for filename in os.listdir(self.folder_path)
            if filename.lower().endswith(VALID_EXTENSIONS)
        ]
        filename_lookup: Dict[str, str] = {}
        for filename in all_filenames:
            lower = filename.lower()
            if lower not in filename_lookup:
                filename_lookup[lower] = filename
            match = pattern.match(filename)
            if not match:
                continue
            base = match.group("base")
            index = int(match.group("index"))
            image_path = os.path.join(self.folder_path, filename)
            file_hash = FileHelper.hash_file(image_path)
            base_key = base.lower()
            group = grouped.get(base_key)
            if group is None:
                group = {"base": base, "files": []}
                grouped[base_key] = group
            group["files"].append((index, filename, image_path, file_hash))

        composite_sets: List[Tuple[str, List[Tuple[int, str, str, str]]]] = []
        for group in grouped.values():
            files = group["files"]
            if not files:
                continue
            base = group["base"]
            base_lower = base.lower()
            for ext in VALID_EXTENSIONS:
                candidate_lower = f"{base_lower}{ext}"
                actual = filename_lookup.get(candidate_lower)
                if not actual:
                    continue
                image_path = os.path.join(self.folder_path, actual)
                file_hash = FileHelper.hash_file(image_path)
                files.append((-1, actual, image_path, file_hash))
            files.sort(key=lambda item: (item[0], item[1].lower()))
            composite_sets.append((group["base"], files))

        composite_sets.sort(key=lambda item: item[0].lower())
        return composite_sets

    @staticmethod
    def _compute_composite_hash(files: List[Tuple[int, str, str, str]]) -> str:
        """Compute a hash from the ordered list of composite file hashes."""
        sha1 = hashlib.sha1()
        for _index, filename, _path, file_hash in files:
            sha1.update(f"{filename}\t{file_hash}\n".encode("utf-8"))
        return sha1.hexdigest()

    def collect_composite_tasks(
        self
    ) -> Tuple[List[Tuple[TSVEntry, List[Tuple[int, str, str, str]], str]], Set[str]]:
        """Collect composite tasks and the filenames they cover."""
        if self.no_composites:
            return [], set()

        tasks = []
        skip_filenames: Set[str] = set()
        composite_sets = self._discover_composite_sets()
        composite_entries = self.tsv_handler.get_composite_entries()
        entry_by_base = {
            self._normalize_base_name(entry.original_filename): entry
            for entry in composite_entries
        }
        discovered_bases: Set[str] = set()

        for base_name, files in composite_sets:
            base_key = base_name.lower()
            discovered_bases.add(base_key)
            entry = entry_by_base.get(base_key)
            if entry is None:
                entry = TSVEntry(
                    original_filename=base_name,
                    short_desc="",
                    long_desc="",
                    context="",
                    composite=True,
                    file_hash=""
                )
                self.tsv_handler.entries.append(entry)
            else:
                entry.composite = True

            for _index, filename, _path, _file_hash in files:
                skip_filenames.add(filename.lower())

            composite_hash = self._compute_composite_hash(files)
            previous_hash = entry.file_hash
            self.tsv_handler.update_entry_hash(entry, composite_hash)
            if entry.short_desc and entry.long_desc and previous_hash == composite_hash:
                continue
            tasks.append((entry, files, composite_hash))

        for entry in composite_entries:
            base_key = self._normalize_base_name(entry.original_filename)
            if base_key not in discovered_bases:
                base_name = Path(entry.original_filename).stem
                logger.warning(f"No composite files found for base name '{base_name}'.")

        return tasks, skip_filenames

    def process_composite_task(
        self,
        task: Tuple[TSVEntry, List[Tuple[int, str, str, str]], str]
    ) -> Optional[CompositeResult]:
        """Process a composite image set and return the result."""
        entry, files, composite_hash = task
        if not self.describer:
            logger.error("Image describer is not initialized.")
            return None
        image_paths = [path for _index, _filename, path, _hash in files]
        short_desc, long_desc = self.describer.describe_images(image_paths, entry.context)
        if short_desc == "Error":
            return None
        return CompositeResult(
            entry=entry,
            composite_hash=composite_hash,
            short_desc=short_desc,
            long_desc=long_desc
        )

    def discover_composites(self) -> List[Tuple[str, List[str]]]:
        """Discover composite sets based on underscore-numbered filenames."""
        composites: List[Tuple[str, List[str]]] = []
        if self.no_composites:
            return composites
        for base_name, files in self._discover_composite_sets():
            composites.append((base_name, [filename for _index, filename, _path, _hash in files]))
        return composites

    def show_composites(self) -> None:
        """Print composite base names and their matching files."""
        if self.no_composites:
            logger.info("Composite detection is disabled.")
            return
        composites = self.discover_composites()
        if not composites:
            logger.info("No composites found.")
            return
        for base_name, files in composites:
            logger.info(f"{base_name}:")
            for filename in files:
                logger.info(f"  - {filename}")

    def handle_composite_result(self, result: CompositeResult) -> None:
        """Update TSV entry for a composite result (no file copying)."""
        short_desc = FileHelper.sanitize_filename(result.short_desc)
        result.entry.short_desc = short_desc
        result.entry.long_desc = result.long_desc
        self.tsv_handler.update_entry_hash(result.entry, result.composite_hash)

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
        composite_tasks, skip_filenames = self.collect_composite_tasks()
        tasks = self.collect_image_files(skip_filenames=skip_filenames)
        total_count = len(tasks) + len(composite_tasks)
        composite_count = len(composite_tasks)
        single_image_count = len(tasks)
        composite_image_count = sum(len(files) for _entry, files, _hash in composite_tasks)
        total_images = single_image_count + composite_image_count
        def _label(count: int, singular: str) -> str:
            return singular if count == 1 else f"{singular}s"

        row_label = _label(total_count, "row") if composite_count else _label(total_count, "image")
        
        if total_count == 0:
            self.tsv_handler.write_all()
            logger.info("No new images to process.")
            return
        
        if composite_count:
            logger.info(
                "Processing "
                f"{total_count} {row_label} "
                f"({single_image_count} {_label(single_image_count, 'single image')}, "
                f"{composite_count} {_label(composite_count, 'composite row')}, "
                f"{total_images} {_label(total_images, 'image')} total)..."
            )
        else:
            logger.info(f"Processing {total_count} {row_label}...")

        # We'll use this to track progress
        done_count = 0
        
        # Parallel describing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_kind = {}
            for task in tasks:
                future_to_kind[executor.submit(self.process_image, task)] = "image"
            for task in composite_tasks:
                future_to_kind[executor.submit(self.process_composite_task, task)] = "composite"
            
            for future in as_completed(future_to_kind):
                result = future.result()
                
                with self.progress_lock:
                    done_count += 1
                    logger.info(f"{done_count} of {total_count} {row_label}")
                
                if result:
                    kind = future_to_kind[future]
                    if kind == "image":
                        self.handle_image_result(result)
                    else:
                        self.handle_composite_result(result)
        
        # Rewrite TSV with updated entries
        self.tsv_handler.write_all()
        logger.info(f"Processed {done_count} {row_label}.")

    def init_tsv(self) -> None:
        """Initialize a TSV with hashes and empty descriptions/context."""
        all_files = sorted(os.listdir(self.folder_path), key=str.lower)
        pending_hashes = set()
        self.tsv_handler.entries = []
        self.tsv_handler.entries_by_hash.clear()

        total_count = 0
        skip_filenames: Set[str] = set()
        if not self.no_composites:
            for base_name, files in self._discover_composite_sets():
                for _index, filename, _path, _file_hash in files:
                    skip_filenames.add(filename.lower())
                composite_hash = self._compute_composite_hash(files)
                total_count += 1
                self.tsv_handler.upsert_entry(
                    orig_filename=base_name,
                    short_desc="",
                    long_desc="",
                    context="",
                    file_hash=composite_hash,
                    composite=True
                )

        for filename in all_files:
            if not filename.lower().endswith(VALID_EXTENSIONS):
                continue
            if filename.lower() in skip_filenames:
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
        logger.info(f"Initialized TSV for {total_count} rows.")


class CLI:
    """Command-line interface handler."""
    
    @staticmethod
    def parse_args() -> argparse.Namespace:
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser(
            description="Describe images with OpenAI. For folders, writes 6 columns in TSV:\n"
                        "OriginalFilename, ShortDescription, LongDescription, Context, Composite, SHA1.\n"
                        "For single image files, outputs descriptions directly."
        )
        parser.add_argument("path", nargs="?", help="Path to folder or single image file.")
        parser.add_argument(
            "-t", "--temperature",
            type=float,
            help="Sampling temperature for OpenAI (default=1.0)."
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
            help="Path to the configuration file (default: config.json in the target folder, current directory, or ~/.config/gid/config.json)"
        )
        parser.add_argument(
            "--init-tsv",
            action="store_true",
            help="Generate TSV with hashes and empty descriptions/context (folder mode only; composites auto-detected unless --no-composites)."
        )
        parser.add_argument(
            "--no-composites",
            action="store_true",
            help="Disable automatic composite detection (process all images individually)."
        )
        parser.add_argument(
            "--show-composites",
            action="store_true",
            help="List discovered composite sets and their matching files (folder mode only)."
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
        config_path = args.config
        if not config_path and args.path:
            target_dir = None
            if os.path.isdir(args.path):
                target_dir = args.path
            elif os.path.isfile(args.path):
                target_dir = os.path.dirname(args.path)
            if target_dir:
                candidate = os.path.join(target_dir, "config.json")
                if os.path.exists(candidate):
                    config_path = candidate
        config = Config.load_config(config_path)
        
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
        if args.no_composites:
            config["processing"]["no_composites"] = True
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
            single_image_prompt=config["prompt"].get("single_image_prompt"),
            composite_image_prompt=config["prompt"].get("composite_image_prompt"),
            context_template=config["prompt"].get("context_template"),
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
        if not args.init_tsv and not args.show_composites and not config["api"]["api_key"]:
            print("Error: OpenAI API key not provided. Use -k/--api-key, set OPENAI_API_KEY environment variable, or add it to config.json.", file=sys.stderr)
            sys.exit(1)
        
        # Check if path is a directory or a file
        if os.path.isdir(args.path):
            if args.show_composites:
                processor = ImageProcessor(
                    folder_path=args.path,
                    config=config,
                    init_only=True
                )
                processor.show_composites()
            elif args.init_tsv:
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
            if args.show_composites:
                print("Error: --show-composites is only supported for folder mode.", file=sys.stderr)
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
