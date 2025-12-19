# GID - Generate Image Descriptions (Agent Notes)

## Purpose
Generate short and long textual descriptions for images using the OpenAI API. The tool can process a single image (stdout only) or a folder (TSV output plus optional copies).

## Repository Layout
- `gid.py`: single entry point, all logic lives here
- `config.json`: repo default config (overrides code defaults when present)
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
- `-n`, `--no-copy`: skip copying in folder mode
- `-w`, `--workers`: max worker threads (folder mode)
- `-v`, `--verbose`: enable OpenAI + HTTPX request logs
- `-c`, `--config`: path to config file

## Configuration Resolution
1. Start from `Config.DEFAULT_CONFIG` in `gid.py` (model `gpt-5.2`, temperature `0.7`, max tokens `4000`).
2. If a config file exists, deep-merge it:
   - `config.json` in the current directory, else `~/.config/gid/config.json`
   - The repo includes `config.json`, which currently matches the defaults but still overrides.
3. CLI flags override config values.
4. `OPENAI_API_KEY` is used only if no API key was provided by file or CLI.
Note: the CLI help text lists defaults (0.7, 4000, gpt-5.2), but the actual values come from the config resolution above.

## Modes and Output
- **Folder mode** (path is a directory):
  - Collects images with extensions: `.png .jpg .jpeg .gif .bmp .tiff .webp`
  - Sorts filenames case-insensitively, but processes results in completion order (not strict input order).
  - Writes `descriptions.tsv` (header only if file does not exist):
    - `OriginalFilename`, `ShortDescription`, `LongDescription`, `SHA1`
  - Uses SHA-1 hashes to skip files already present in the TSV and to skip duplicates within the same run.
  - Copies images into `Described/` by default; `--no-copy` keeps everything in the source folder.
  - Short description is sanitized for filename safety; collisions are resolved with `" 2"`, `" 3"`, ... up to 100.
- **Single image mode** (path is a file):
  - Outputs short description, then long description to stdout.
  - No files are created.

## OpenAI Call Behavior (Important)
- Uses the Responses API with `input_image` content and `instructions` for the system prompt.
- Sends `temperature=<config>` and `max_output_tokens=<max_tokens>`.
- Short descriptions are trimmed to `short_description_max_words` (default 10).
- The prompt instructs the model to output:
  - short description, newline, long description
- The response parser expects:
  - `SHORT: ...` and `LONG: ...`
  - If not found, it falls back to the first N words (default 10) as the short description.

## Code Style Notes
- Standard library imports first, then third-party, then local
- 4-space indentation; partial type hints via `typing`
- Uses `ThreadPoolExecutor` for parallel describing
- Log output is minimal by default; `--verbose` enables HTTP request logs
