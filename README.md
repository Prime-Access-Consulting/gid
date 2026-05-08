# GID - Generate Image Descriptions

GID is a Python tool for automatically generating human-readable descriptions of images using a configured OpenAI model. It processes image files in a specified folder, creates both short and detailed descriptions, and optionally renames and copies the files based on their descriptions.

## Features

- **Dual Descriptions**: Generates both short (filename-friendly) and detailed descriptions
- **Flexible Usage Modes**:
  - Process a single image with direct console output
  - Process entire folders of images with organized storage
- **Efficient Processing**: Processes images in parallel using multiple threads
- **Deduplication**: Uses SHA-1 hashing to avoid reprocessing duplicate images
- **Organized Output**: 
  - Stores all descriptions in a TSV file for easy reference (folder mode)
  - Optionally copies images to a "Described" subfolder with descriptive filenames (folder mode)
- **Progress Tracking**: Shows processing progress as you go
- **Resumable**: Can be stopped and restarted without duplicating work
- **Optional Composite Support**: Detect multi-image composites by filename and describe them as a single unit when explicitly enabled

## Installation

### Prerequisites

- Python 3.7+
- OpenAI API key with access to the configured model. The default is `gpt-5.5`.

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/prime-access-consulting/gid.git
   cd gid
   ```

2. Install dependencies:
   ```bash
   python3 -m pip install -r requirements.txt
   ```

   On Windows, use `py -3` or `python` if that command points to Python 3.

3. Set your OpenAI API key (choose one method):

   a. Using environment variable:
   ```bash
   # Linux/macOS
   export OPENAI_API_KEY=your_api_key_here
   
   # Windows (Command Prompt)
   set OPENAI_API_KEY=your_api_key_here
   
   # Windows (PowerShell)
   $env:OPENAI_API_KEY="your_api_key_here"
   ```
   
   b. Using command-line argument:
   ```bash
   python3 gid.py /path/to/images -k your_api_key_here
   ```

## Usage

### Basic Usage

#### Process a Folder of Images

```bash
python3 gid.py /path/to/your/images
```

This will:
1. Process all images in the specified folder
2. Create a "Described" subfolder
3. Generate a `descriptions.tsv` file with all descriptions
4. Copy processed images to the "Described" subfolder with descriptive filenames

#### Process a Single Image

```bash
python3 gid.py /path/to/your/image.jpg
```

This will:
1. Process the single image file
2. Output the short description and long description directly to the console
3. No files are created or copied

### Command-Line Options

```
usage: gid.py [-h] [-t TEMPERATURE] [-l LENGTH] [-n] [-k API_KEY] [-w WORKERS]
              [-v] [-c CONFIG] [--init-tsv] [--force-init-tsv] [--make-excel]
              [--composites | --no-composites] [--show-composites]
              [--write-sample-config [PATH]] [-m MODEL] [-p NAME]
              [--reasoning-effort {none,low,medium,high,xhigh} | --no-reasoning]
              [path]

positional arguments:
  path                  Path to folder or single image file.

options:
  -h, --help            show this help message and exit
  -t, --temperature TEMPERATURE
                        Sampling temperature for OpenAI (default=1.0).
  -l, --length LENGTH   Max tokens (default=4000).
  -n, --no-copy         If provided, do NOT copy files to the output folder (folder mode only).
  -k, --api-key API_KEY
                        OpenAI API key (overrides config file and environment variable).
  -w, --workers WORKERS
                        Maximum number of concurrent workers in folder mode (0=auto, default=0).
  -v, --verbose         Enable verbose output including HTTP requests.
  -c, --config CONFIG
                        Path to the configuration file (default: config.json in the target folder, current directory, or ~/.config/gid/config.json)
  --init-tsv            Generate TSV with hashes and empty descriptions/context (folder mode only; use --composites to include composite rows).
  --force-init-tsv      Reset the TSV when using --init-tsv instead of preserving existing rows/context.
  --make-excel          Generate an Excel .xlsx file from the existing TSV (folder mode only).
  --composites          Enable automatic composite detection (folder mode only).
  --no-composites       Disable automatic composite detection (default; useful to override config).
  --show-composites     List discovered composite sets and their matching files (folder mode only; no API calls or output writes).
  --write-sample-config [PATH]
                        Write built-in defaults to a sample config file and exit (default: config.json.sample).
  -m, --model MODEL     OpenAI model ID to send to the API (default: gpt-5.5).
  -p, --prompt NAME     System prompt file name from a prompts/ directory (for example: web for prompts/web.md).
  --reasoning-effort {none,low,medium,high,xhigh}
                        Reasoning effort for supported models (default: medium).
                        Choices: none, low, medium, high, xhigh.
  --no-reasoning        Do not send a reasoning parameter; use this for models that do not support reasoning.
```

Temperature defaults to **1.0** unless you override it with `--temperature` or the config file.

### Examples

Process images with higher temperature (more creative descriptions):
```bash
python3 gid.py /path/to/images --temperature 0.9
```

Process images but don't copy them (just create descriptions.tsv):
```bash
python3 gid.py /path/to/images --no-copy
```

Process images with longer descriptions:
```bash
python3 gid.py /path/to/images --length 1200
```

Process images with a specific number of concurrent workers:
```bash
python3 gid.py /path/to/images --workers 4
```

Process a single image with higher temperature:
```bash
python3 gid.py /path/to/image.jpg --temperature 0.9
```

Use a higher reasoning effort with the default reasoning model:
```bash
python3 gid.py /path/to/images --reasoning-effort high
```

Disable the reasoning parameter for a model that does not support it:
```bash
python3 gid.py /path/to/images --model some-model --no-reasoning
```

Use a specific model ID:
```bash
python3 gid.py /path/to/images --model gpt-5.5
```

Use a system prompt from `prompts/web.md`:
```bash
python3 gid.py /path/to/images --prompt web
```

Enable verbose mode to see detailed API request logs:
```bash
python3 gid.py /path/to/images --verbose
```

Use a specific config file:
```bash
python3 gid.py /path/to/images --config /path/to/my-config.json
```

Generate an Excel file from an existing TSV (no API calls):
```bash
python3 gid.py /path/to/images --make-excel
```

Regenerate the sample config from built-in defaults:
```bash
python3 gid.py --write-sample-config
```

### Configuration File

You can use a JSON configuration file to customize GID's behavior. The config file can be placed in one of these locations:
- `config.json` in the folder you are describing
- `config.json` in the current directory
- `~/.config/gid/config.json` in your home directory
- Any custom path specified with the `--config` flag

Defaults are defined centrally in `gid.py`. A sample configuration file is provided as `config.json.sample`, but the running program does not depend on it. Your own `config.json` only needs values you want to override. Regenerate the sample at any time with `python3 gid.py --write-sample-config`. Placeholder values such as `"api_key": "..."` are for documentation and are treated as unset.

The configuration file supports the following settings:

```json
{
  "api": {
    "api_key": "...",       // Optional; prefer OPENAI_API_KEY or --api-key
    "model": "gpt-5.5"     // OpenAI model ID to send to the API
  },
  "parameters": {
    "temperature": 1.0,    // Sampling temperature
    "max_tokens": 4000,    // Maximum response tokens
    "reasoning_effort": "medium" // none, low, medium, high, xhigh, or null to omit reasoning
  },
  "processing": {
    "no_copy": false,      // Whether to skip copying files
    "no_composites": true, // Disable automatic composite detection
    "max_workers": 0,      // Max concurrent workers (0 = auto)
    "verbose": false       // Enable verbose logging
  },
  "output": {
    "output_folder_name": "Described",  // Name of output folder
    "tsv_filename": "descriptions.tsv"  // Name of TSV file
  },
  "prompt": {
    "system_prompt": "default", // Built-in prompts/default.md
    "instructions_prompt": "instructions", // Built-in prompts/instructions.md placed before system_prompt
    "single_image_prompt": "Describe the following image using the required SHORT/LONG output format.",
    "composite_image_prompt": "Describe the following images together as a single composite using the required SHORT/LONG output format.",
    "context_template": "Additional image facts provided by the user (treat as true and naturally incorporate that knowledge if helpful or necessary to inform the description): {context}",
    "short_description_max_words": 10   // Max words in short description
  }
}
```

Command-line arguments will override settings in the configuration file.
Use `{context}` in `context_template`; it will be replaced with the row's **Context** value.
Use `{short_description_max_words}` in prompt text when referring to the configured short-description length, so changing the numeric value updates the rendered prompt.
Model IDs are passed directly to the OpenAI API. GID does not resolve aliases such as `latest` or `5`.
The default `reasoning_effort` is `medium`; accepted values are `none`, `low`, `medium`, `high`, and `xhigh`. Set it to `null` in config or pass `--no-reasoning` to omit the API `reasoning` parameter entirely.

Prompt fields can use inline prompt text or prompt file references. A bare path-like value such as `web`, `Brief`, `web.md`, or `prompts/web.md` is treated as a prompt file reference, not as inline prompt text; GID must find the referenced Markdown prompt or it exits with an error. Prompt directories are searched next to the target folder, next to the active config file, next to the script for bundled prompts, in the current directory, and in `~/.config/gid/prompts`. The `-p/--prompt` flag is a shortcut for overriding `prompt.system_prompt` with one of those prompt files. `prompt.instructions_prompt` is placed before the selected system prompt so reusable output-format instructions stay prominent across prompt variants. Inline prompt text should be written as the actual prompt prose, usually a full sentence or paragraph.

### Prompt Output Format

The bundled instructions prompt asks the model to return two labeled fields:

- `SHORT: <short description>`
- `LONG: <long description>`

GID strips these labels when saving to the TSV. The long description may include paragraph breaks after the `LONG:` label. If the model returns a malformed response, GID rejects that result instead of inventing a filename from the long description.

## Output Format

### TSV File

The script generates a tab-separated values (TSV) file with the following columns:
1. **OriginalFilename**: The original filename of the image
2. **ShortDescription**: A short description suitable for filenames, trimmed to `short_description_max_words`
3. **LongDescription**: A detailed description of the image content
4. **Context**: Optional per-image facts provided by a user to improve descriptions
5. **Composite**: `yes` or `no` (case-insensitive). If `yes`, this row represents a composite image set (auto-detected by filename only when composites are enabled).
6. **SHA1**: A SHA-1 hash of the image file for deduplication

The TSV is written as plain UTF-8 with one physical row per image. Newlines inside descriptions/context are stored as literal `\n` sequences so spreadsheet apps keep the row and columns stable. Common smart punctuation is normalized to plain ASCII punctuation. Use `--make-excel` when you need a spreadsheet with real multiline cells.

### Excel Output

Use `--make-excel` to generate `descriptions.xlsx` in the same folder as the TSV. The spreadsheet uses the same columns, with unescaped text (real newlines and Unicode preserved).

### Initialize a TSV for Context

To create a TSV with hashes and empty description fields (so someone can fill in Context), run:

```bash
python3 gid.py /path/to/images --init-tsv
```

This does not call the API or copy any files. Then add per-image context in the **Context** column and rerun the tool normally. If **ShortDescription** or **LongDescription** is empty or appears malformed, GID will generate descriptions for that row and append the context to the prompt as additional image facts. The **Composite** column is set to `yes` for detected composite rows and `no` for single-image rows.
Composite detection is disabled by default. Use `--composites` with `--init-tsv` to add composite rows based on the `base_<number>.<ext>` filename pattern.
By default, rerunning `--init-tsv` preserves existing rows, context, and descriptions when hashes still match. If a file's hash changes but the filename or composite base still matches, GID preserves the context and clears descriptions so the row will be regenerated. Use `--force-init-tsv` with `--init-tsv` to reset the TSV instead.

### Composite Images

To describe multiple related images as a single composite:

1. Name the files using the `base_<number>.<ext>` pattern (for example, `sina_1.jpg`, `sina_2.jpg`, `sina_3.jpg`).
2. (Optional) If you also have a `base.<ext>` file (for example, `sina.jpg`), it will be included in the composite set automatically.
3. (Optional) Run `--init-tsv --composites` to generate a composite row, then add any shared **Context** to that row.
4. Run GID with `--composites`.

When GID runs, it will find all matching `base_<number>.<ext>` files (plus `base.<ext>` if present), send them together in one request, and save the composite description on the composite row. Component rows are skipped during processing. The composite row’s **OriginalFilename** includes an extension (for example, `base.jpg`) to preserve naming. The composite row's **SHA1** is computed from the ordered list of component filenames + hashes, so changing any component triggers a reprocess.

### Described Folder

Unless `--no-copy` is specified, processed images are copied to a "Described" subfolder with filenames based on their short descriptions.

## License

[Creative Commons Attribution-NonCommercial 4.0 International License](LICENSE)

This project is licensed for non-commercial use only. If you'd like to use this project for commercial purposes, please contact the author.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

This tool uses OpenAI's API to generate image descriptions.
