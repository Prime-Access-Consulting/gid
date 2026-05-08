# GID - Generate Image Descriptions

GID is a Python CLI for generating short, filename-friendly image descriptions and longer human-readable descriptions with an OpenAI model. It can process one image, one folder, or many folders recursively, and stores folder results in a resumable TSV workflow with optional Excel export.

## What It Does

- Generates both short and long descriptions for image files.
- Processes a single image to stdout, a folder to TSV, or matching folders recursively.
- Writes `descriptions.tsv` with original filename, descriptions, optional context, composite status, and SHA-1 hash.
- Preserves work between runs by using SHA-1 hashes and regenerating only missing or malformed descriptions.
- Supports a context-first workflow with `--init-tsv`, so users can add per-image facts before generation.
- Optionally copies described images into a `Described/` folder using sanitized short descriptions as filenames.
- Can skip copying with `--no-copy` and keep all output in the source folder.
- Can export an existing TSV to `descriptions.xlsx` with `--make-excel`.
- Supports prompt files, custom config files, model overrides, reasoning controls, and temperature/max-token settings.
- Supports optional composite detection for filename sequences such as `base_1.jpg`, `base_2.jpg`, `base_3.jpg`.

## Installation

### Prerequisites

- Python 3.7+
- An OpenAI API key with access to the configured model. The default model ID is `gpt-5.5`.

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/prime-access-consulting/gid.git
   cd gid
   ```

2. Install dependencies:

   ```bash
   python3 -m pip install -r requirements.txt
   ```

   On Windows, use `py -3` or `python` if that command points to Python 3.

3. Set your API key with one of these methods:

   ```bash
   # Linux/macOS
   export OPENAI_API_KEY=your_api_key_here

   # Windows Command Prompt
   set OPENAI_API_KEY=your_api_key_here

   # Windows PowerShell
   $env:OPENAI_API_KEY="your_api_key_here"
   ```

   You can also pass a key per command:

   ```bash
   python3 gid.py /path/to/images --api-key your_api_key_here
   ```

## Quick Start

Process a folder of images:

```bash
python3 gid.py /path/to/images
```

By default, folder mode creates `Described/descriptions.tsv` and copies processed images into `Described/` with filenames based on their short descriptions.

Process a folder without copying images:

```bash
python3 gid.py /path/to/images --no-copy
```

With `--no-copy`, GID writes `descriptions.tsv` directly in the source folder.

Process one image:

```bash
python3 gid.py /path/to/image.jpg
```

Single-image mode prints the short description and long description to stdout. It does not write files.

Recursively process every image-containing folder below the current directory:

```bash
python3 gid.py --recurse
```

Recursively process only folders named `final` below the current directory:

```bash
python3 gid.py --recurse final
```

## Common Workflows

### Add Context Before Describing

Create a TSV with image hashes and empty description/context fields:

```bash
python3 gid.py /path/to/images --init-tsv
```

Fill in the `Context` column with per-image facts, then run GID normally:

```bash
python3 gid.py /path/to/images
```

`--init-tsv` does not call the API. By default, rerunning it preserves existing rows, context, and descriptions when hashes still match. If a file changes under the same filename or composite base, GID preserves context and clears descriptions so the row can be regenerated. Use `--force-init-tsv` to reset the TSV.

### Export Excel

Generate an Excel file from an existing TSV:

```bash
python3 gid.py /path/to/images --make-excel
```

This does not call the API. It writes `descriptions.xlsx` next to the TSV.

### Tune Generation

Use a specific model:

```bash
python3 gid.py /path/to/images --model gpt-5.5
```

Adjust temperature:

```bash
python3 gid.py /path/to/images --temperature 0.7
```

Temperature defaults to `1.0`. GID only sends `temperature` when it differs from the default; if the selected model rejects temperature, GID exits with a clear error instead of continuing through the batch.

Adjust max output tokens:

```bash
python3 gid.py /path/to/images --length 1200
```

Set reasoning effort for models that support it:

```bash
python3 gid.py /path/to/images --reasoning-effort high
```

Omit the reasoning parameter for models that do not support it:

```bash
python3 gid.py /path/to/images --model some-model --no-reasoning
```

### Use Prompt Files

Use `prompts/web.md` as the system prompt:

```bash
python3 gid.py /path/to/images --prompt web
```

Prompt references such as `web`, `web.md`, and `prompts/web.md` refer to Markdown prompt files. Missing referenced prompt files are fatal errors.

### Composite Images

Composite processing is disabled by default. Enable it when files represent one logical image set:

```bash
python3 gid.py /path/to/images --composites
```

GID detects files named with the `base_<number>.<ext>` pattern, such as `sina_1.jpg`, `sina_2.jpg`, and `sina_3.jpg`. If `sina.jpg` also exists, it is included in the composite set. Component rows are skipped during processing, and one composite row receives the generated description.

To add shared context first:

```bash
python3 gid.py /path/to/images --init-tsv --composites
```

Then add context to the composite row and rerun with `--composites`.

## Recursive Processing

`--recurse` always starts from the current directory.

With no positional value, GID processes every folder below the current directory that directly contains image files:

```bash
python3 gid.py --recurse
```

With a positional value, it must be a bare folder name filter:

```bash
python3 gid.py --recurse final
```

Path-like values such as `.` or `./photos` are rejected in recursive mode. To recurse from another location, change into that directory first and run `python3 /path/to/gid.py --recurse`.

Recursive mode skips generated `Described` folders, hidden folders, source-control folders, and common environment/cache folders. Other CLI options are applied to every matched folder. Folder-local `config.json` files and `prompts/` directories are resolved per matched folder unless `--config` is supplied.

## Configuration

GID starts from defaults defined in `gid.py`, then merges config file values, then applies CLI overrides.

If `--config` is supplied, GID uses that file for every target and the file must exist. Otherwise, config files are searched in this order:

1. `config.json` in the target folder
2. `config.json` in the current directory
3. `~/.config/gid/config.json`

Placeholder API keys such as `"..."` are treated as unset, so `OPENAI_API_KEY` can still provide the key.

The config file is strict JSON. Comments are not allowed in the file itself.

```json
{
  "api": {
    "api_key": "...",
    "model": "gpt-5.5"
  },
  "parameters": {
    "temperature": 1.0,
    "max_tokens": 4000,
    "reasoning_effort": "medium"
  },
  "processing": {
    "no_copy": false,
    "no_composites": true,
    "max_workers": 0,
    "verbose": false
  },
  "output": {
    "output_folder_name": "Described",
    "tsv_filename": "descriptions.tsv"
  },
  "prompt": {
    "system_prompt": "default",
    "instructions_prompt": "instructions",
    "single_image_prompt": "Describe the following image using the required SHORT/LONG output format.",
    "composite_image_prompt": "Describe the following images together as a single composite using the required SHORT/LONG output format.",
    "context_template": "Additional image facts provided by the user (treat as true and naturally incorporate that knowledge if helpful or necessary to inform the description): {context}",
    "short_description_max_words": 10
  }
}
```

Key settings:

| Setting | Notes |
| --- | --- |
| `api.api_key` | Optional; prefer `OPENAI_API_KEY` or `--api-key` for secrets. |
| `api.model` | Model ID passed directly to the OpenAI API. GID does not resolve aliases such as `latest` or `5`. |
| `parameters.temperature` | Sampling temperature. Not every model supports this parameter. |
| `parameters.max_tokens` | Maximum response tokens, mapped to `max_output_tokens`. |
| `parameters.reasoning_effort` | `none`, `low`, `medium`, `high`, `xhigh`, or `null` to omit reasoning. |
| `processing.no_copy` | `true` writes TSV output in the source folder instead of `Described/`. |
| `processing.no_composites` | `true` disables automatic composite detection. |
| `processing.max_workers` | Maximum worker threads in folder mode; `0` means auto. |
| `output.output_folder_name` | Folder used for copied output and TSVs when copying is enabled. |
| `output.tsv_filename` | TSV filename. |
| `prompt.short_description_max_words` | Maximum words in the short description. |

Regenerate the sample config from built-in defaults:

```bash
python3 gid.py --write-sample-config
```

## Prompt Behavior

Prompt fields can be inline prompt text or prompt file references. Bare path-like values such as `web`, `Brief`, `web.md`, and `prompts/web.md` are treated as prompt file references, not inline prompt text. Inline prompt text should be actual prompt prose, usually a full sentence or paragraph.

Prompt directories are searched in this order:

1. `prompts/` next to the target folder
2. `prompts/` next to the active config file
3. Bundled `prompts/` next to `gid.py`
4. `prompts/` in the current directory
5. `~/.config/gid/prompts`

`--prompt NAME` overrides `prompt.system_prompt` with one of those prompt files. `prompt.instructions_prompt` is placed before the selected system prompt so reusable output-format instructions stay prominent across prompt variants.

Use `{context}` in `context_template`; it is replaced with the row's `Context` value. Use `{short_description_max_words}` in prompt text when referring to the configured short-description length.

The bundled instructions prompt asks the model to return two labeled fields:

```text
SHORT: <short description>
LONG: <long description>
```

GID strips these labels when saving to the TSV. Long descriptions are saved as one plain-text paragraph. If the model returns a malformed response, GID rejects that result instead of inventing a filename from the long description.

## Output Format

### TSV

Folder mode writes a tab-separated file with these columns:

1. `OriginalFilename`: Original image filename.
2. `ShortDescription`: Short description suitable for filenames, trimmed to `short_description_max_words`.
3. `LongDescription`: Detailed description of the image content.
4. `Context`: Optional per-image facts supplied by a user.
5. `Composite`: `yes` or `no`; `yes` means the row represents a composite image set.
6. `SHA1`: SHA-1 hash used for deduplication and resumability.

The TSV is plain UTF-8 with one physical row per image. Long descriptions are collapsed to one plain-text paragraph. Newlines inside context are stored as literal `\n` sequences so spreadsheet apps keep rows and columns stable. Common smart punctuation is normalized to ASCII punctuation.

If `ShortDescription` or `LongDescription` is empty or appears malformed, including generic long-description openings such as "The image shows", GID will regenerate that row.

### Excel

`--make-excel` writes `descriptions.xlsx` next to the TSV. The spreadsheet uses the same columns, restores context newlines, preserves Unicode, and keeps long descriptions as one paragraph.

### Described Folder

Unless `--no-copy` is set, processed images are copied to `Described/` with filenames based on their short descriptions. Filename collisions are resolved with numeric suffixes.

## CLI Reference

```text
usage: gid.py [-h] [--recurse] [-t TEMPERATURE] [-l LENGTH] [-n] [-k API_KEY]
              [-w WORKERS] [-v] [-c CONFIG] [--init-tsv] [--force-init-tsv]
              [--make-excel] [--composites | --no-composites]
              [--show-composites] [--write-sample-config [PATH]] [-m MODEL]
              [-p NAME] [--reasoning-effort {none,low,medium,high,xhigh} |
              --no-reasoning]
              [path]

positional arguments:
  path                  Path to folder or single image file. With --recurse,
                        this must be a bare folder name to find.

options:
  -h, --help            show this help message and exit
  --recurse             Recursively process image-containing folders from the
                        current directory. An optional positional value
                        filters by folder name.
  -t, --temperature TEMPERATURE
                        Sampling temperature for OpenAI (default=1.0).
  -l, --length LENGTH   Max tokens (default=4000).
  -n, --no-copy         If provided, do NOT copy files to the output folder
                        (folder mode only).
  -k, --api-key API_KEY
                        OpenAI API key (overrides config file and environment
                        variable).
  -w, --workers WORKERS
                        Maximum number of concurrent workers in folder mode
                        (0=auto, default=0).
  -v, --verbose         Enable verbose output including HTTP requests.
  -c, --config CONFIG   Path to the configuration file (default: config.json
                        in the target folder, current directory, or
                        ~/.config/gid/config.json)
  --init-tsv            Generate TSV with hashes and empty
                        descriptions/context (folder mode only; use
                        --composites to include composite rows).
  --force-init-tsv      Reset the TSV when using --init-tsv instead of
                        preserving existing rows/context.
  --make-excel          Generate an Excel .xlsx file from the existing TSV
                        (folder mode only).
  --composites          Enable automatic composite detection (folder mode
                        only).
  --no-composites       Disable automatic composite detection (default; useful
                        to override config).
  --show-composites     List discovered composite sets and their matching
                        files (folder mode only; no API calls or output
                        writes).
  --write-sample-config [PATH]
                        Write built-in defaults to a sample config file and
                        exit (default: config.json.sample).
  -m, --model MODEL     OpenAI model ID to send to the API (default: gpt-5.5).
  -p, --prompt NAME     System prompt file name from a prompts/ directory (for
                        example: web for prompts/web.md).
  --reasoning-effort {none,low,medium,high,xhigh}
                        Reasoning effort for supported models (default:
                        medium). Choices: none, low, medium, high, xhigh.
  --no-reasoning        Do not send a reasoning parameter; use this for models
                        that do not support reasoning.
```

## License

[Creative Commons Attribution-NonCommercial 4.0 International License](LICENSE)

This project is licensed for non-commercial use only. If you'd like to use this project for commercial purposes, please contact the author.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

This tool uses OpenAI's API to generate image descriptions.
