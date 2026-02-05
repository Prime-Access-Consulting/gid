# GID - Generate Image Descriptions (Agent Notes)

## Purpose
Generate short and long textual descriptions for images using the OpenAI API. The tool can process a single image (stdout only) or a folder (TSV output plus optional copies).

## Repository Layout
- `gid.py`: single entry point, all logic lives here
- `config.json.sample`: sample config (copy to `config.json` to override defaults)
- `requirements.txt`: depends on `openai>=1.0.0`
- `README.md`: user-facing docs

## Run Commands
```bash
# Process a folder of images (creates TSV and optional copies)
python gid.py /path/to/images

# Process a single image (outputs descriptions to console only)
python gid.py /path/to/image.jpg

# Override config values
python gid.py /path/to/images --temperature 0.8 --length 1000 --no-copy

# Use a specific config file
python gid.py /path/to/images --config /path/to/config.json

# Select a different model
python gid.py /path/to/images --model gpt-5.2

# Verbose logging (OpenAI + HTTPX request logs)
python gid.py /path/to/images --verbose
```

## CLI Flags
- `path` (positional): folder or image file path
- `-k`, `--api-key`: OpenAI API key
- `-m`, `--model`: OpenAI model name
- `-t`, `--temperature`: sampling temperature
- `-l`, `--length`: max tokens (mapped to `max_output_tokens`)
- `--init-tsv`: generate TSV with hashes and empty descriptions/context (folder mode only, no API calls)
- `--make-excel`: generate an Excel .xlsx from the existing TSV (folder mode only, no API calls)
- `--no-composites`: disable automatic composite detection
- `--show-composites`: list detected composite sets and their matching files (folder mode only)
- `-n`, `--no-copy`: skip copying in folder mode
- `-w`, `--workers`: max worker threads (folder mode)
- `-v`, `--verbose`: enable OpenAI + HTTPX request logs
- `-c`, `--config`: path to config file

## Configuration Resolution
1. Start from `Config.DEFAULT_CONFIG` in `gid.py` (model `gpt-5.2`, temperature `1.0`, max tokens `4000`).
2. If a config file exists, deep-merge it:
   - `config.json` in the folder being described, else current directory, else `~/.config/gid/config.json`
   - The repo includes `config.json.sample` as a starting point.
3. CLI flags override config values.
4. `OPENAI_API_KEY` is used only if no API key was provided by file or CLI.
Note: the CLI help text lists defaults (1.0, 4000, gpt-5.2), but the actual values come from the config resolution above.

## Modes and Output
- **Folder mode** (path is a directory):
  - Collects images with extensions: `.png .jpg .jpeg .gif .bmp .tiff .webp`
  - Sorts filenames case-insensitively, but processes results in completion order (not strict input order).
- Writes `descriptions.tsv` (header only if file does not exist):
    - `OriginalFilename`, `ShortDescription`, `LongDescription`, `Context`, `Composite`, `SHA1`
  - Uses SHA-1 hashes to skip files already present in the TSV and to skip duplicates within the same run.
  - If `ShortDescription` or `LongDescription` is empty for a hash, it will be reprocessed to fill in descriptions.
  - Copies images into `Described/` by default; `--no-copy` keeps everything in the source folder.
  - Short description is sanitized for filename safety; collisions are resolved with `" 2"`, `" 3"`, ... up to 100.
  - Composite detection is automatic (disable with `--no-composites`):
    - Files named like `base_<number>.<ext>` are grouped into one composite set.
    - If `base.<ext>` exists, it is included in the composite set automatically.
    - The composite row uses `OriginalFilename` with an extension (e.g., `sina.jpg`).
    - All composite files are sent together in one request; component rows are skipped.
    - The composite row's SHA-1 is computed from ordered filenames + file hashes, so changes reprocess.
- **Single image mode** (path is a file):
  - Outputs short description, then long description to stdout.
  - No files are created.

## OpenAI Call Behavior (Important)
- Uses the Responses API with `input_image` content and `instructions` for the system prompt.
- Sends `max_output_tokens=<max_tokens>` and includes `temperature` only when it differs from the default 1.0.
- If a row has `Context`, it is appended to the prompt as additional image facts (treated as true).
- Composite rows use a multi-image prompt: "Describe the following images together as a single composite."
- Short descriptions are trimmed to `short_description_max_words` (default 10).
- The prompt instructs the model to output:
  - `SHORT: ...` on line 1
  - `LONG: ...` on line 2
- The response parser accepts:
  - `SHORT:` and `LONG:` labels (with common punctuation like `:` or `-`)
  - Two-line output (first line short, remaining lines long)
  - Fallback: first N words (default 10) as the short description

## Code Style Notes
- Standard library imports first, then third-party, then local
- 4-space indentation; partial type hints via `typing`
- Uses `ThreadPoolExecutor` for parallel describing
- Log output is minimal by default; `--verbose` enables HTTP request logs
