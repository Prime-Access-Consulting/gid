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
python gid.py /path/to/images --model gpt-5.5

# Select reasoning effort for supported reasoning models
python gid.py /path/to/images --reasoning-effort high

# Select the system prompt from prompts/web.md
python gid.py /path/to/images --prompt web

# Enable composite detection
python gid.py /path/to/images --composites

# Verbose logging (OpenAI + HTTPX request logs)
python gid.py /path/to/images --verbose
```

## CLI Flags
- `path` (positional): folder or image file path
- `-k`, `--api-key`: OpenAI API key
- `-m`, `--model`: OpenAI model ID, passed directly to the API
- `-p`, `--prompt`: system prompt file stem from a searched `prompts/` directory, e.g. `web` for `prompts/web.md`
- `--reasoning-effort`: reasoning effort for supported models; choices are `none`, `low`, `medium`, `high`, `xhigh`
- `-t`, `--temperature`: sampling temperature
- `-l`, `--length`: max tokens (mapped to `max_output_tokens`)
- `--init-tsv`: generate TSV with hashes and empty descriptions/context (folder mode only, no API calls)
- `--force-init-tsv`: reset the TSV during `--init-tsv` instead of preserving existing rows/context
- `--make-excel`: generate an Excel .xlsx from the existing TSV (folder mode only, no API calls)
- `--composites`: enable automatic composite detection (off by default)
- `--no-composites`: disable automatic composite detection (default; useful to override config)
- `--show-composites`: list detected composite sets and their matching files (folder mode only, no API calls)
- `--write-sample-config [PATH]`: write built-in defaults to a sample config file and exit (default `config.json.sample`)
- `-n`, `--no-copy`: skip copying in folder mode
- `-w`, `--workers`: max worker threads in folder mode (`0` = auto)
- `-v`, `--verbose`: enable OpenAI + HTTPX request logs
- `-c`, `--config`: path to config file

## Configuration Resolution
1. Start from `Config.DEFAULT_CONFIG` in `gid.py`, which defines the real defaults (model `gpt-5.5`; temperature `1.0`; max tokens `4000`; reasoning effort `medium`; composites disabled; max workers `0`; prompt defaults).
2. If a user config file exists, deep-merge it over the code defaults:
   - `config.json` in the folder being described, else current directory, else `~/.config/gid/config.json`
3. CLI flags override config values.
4. `OPENAI_API_KEY` is used only if no API key was provided by file or CLI.
Note: local `config.json` should contain only values that need overriding. `config.json.sample` mirrors the code defaults plus documentation placeholders such as `"api_key": "..."`; runtime does not depend on it and it can be regenerated with `--write-sample-config`. Placeholder API keys are treated as unset. Model IDs are passed directly to the OpenAI API; GID does not resolve aliases such as `latest` or `5`. Use `{short_description_max_words}` in prompt text rather than duplicating the numeric word limit.
Prompt text fields can be inline strings or prompt file stems. A value such as `web` loads `prompts/web.md`. Prompt directories are searched next to the target folder, next to the active config file, next to the script for bundled prompts, in the current directory, and in `~/.config/gid/prompts`. `-p/--prompt` overrides `prompt.system_prompt` with one of these prompt files. `prompt.instructions_prompt` is placed before the selected system prompt.

## Modes and Output
- **Folder mode** (path is a directory):
  - Collects images with extensions: `.png .jpg .jpeg .gif .bmp .tiff .webp`
  - Sorts filenames case-insensitively, but processes results in completion order (not strict input order).
  - Writes `descriptions.tsv` (header only if file does not exist):
    - `OriginalFilename`, `ShortDescription`, `LongDescription`, `Context`, `Composite`, `SHA1`
  - TSV files are plain UTF-8 with one physical row per image.
  - Newlines inside descriptions/context are stored as literal `\n` sequences so spreadsheet apps keep rows and columns stable.
  - Common smart punctuation is normalized to ASCII punctuation in TSV text fields.
  - Use `--make-excel` for a spreadsheet with real multiline cells.
  - Uses SHA-1 hashes to skip files already present in the TSV and to skip duplicates within the same run.
  - If `ShortDescription` or `LongDescription` is empty or appears malformed for a hash, it will be reprocessed to fill in descriptions.
  - `--init-tsv` preserves existing matching rows by default; if content changes under the same filename/base, it preserves context but clears descriptions. `--force-init-tsv` resets rows.
  - Copies images into `Described/` by default; `--no-copy` keeps everything in the source folder.
  - Short description is sanitized for filename safety; collisions are resolved with `" 2"`, `" 3"`, ... up to 100.
  - Composite detection is disabled by default (enable with `--composites`):
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
- Sends `reasoning={"effort": <reasoning_effort>}` from config/CLI. The default is `medium`.
- If a row has `Context`, it is appended to the prompt as additional image facts (treated as true).
- Composite rows use `prompt.composite_image_prompt` from config.
- Short descriptions are trimmed to `short_description_max_words` (default 10).
- The bundled instructions prompt instructs the model to output:
  - `SHORT: ...` on line 1
  - `LONG: ...` on line 2
- The response parser accepts:
  - `SHORT:` and `LONG:` labels (with common punctuation like `:` or `-`)
  - Legacy two-line output only when the first line plausibly fits the short-description field
  - Malformed responses are rejected instead of using the first words of the long description as the short description

## Code Style Notes
- Standard library imports first, then third-party, then local
- 4-space indentation; partial type hints via `typing`
- Uses `ThreadPoolExecutor` for parallel describing
- Log output is minimal by default; `--verbose` enables HTTP request logs
