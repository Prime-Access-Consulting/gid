# GID - Generate Image Descriptions

GID is a Python tool for automatically generating human-readable descriptions of images using OpenAI's GPT-5.2 model. It processes image files in a specified folder, creates both short and detailed descriptions, and optionally renames and copies the files based on their descriptions.

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
- **Composite Support**: Describe multi-image composites as a single unit with a dedicated TSV row

## Installation

### Prerequisites

- Python 3.7+
- OpenAI API key with access to GPT-5.2

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/prime-access-consulting/gid.git
   cd gid
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

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
   
   b. Using config.json in the current directory:
   ```json
   {
     "api": {
       "api_key": "your_api_key_here"
     }
   }
   ```
   
   c. Using command-line argument:
   ```bash
   python gid.py /path/to/images -k your_api_key_here
   ```

## Usage

### Basic Usage

#### Process a Folder of Images

```bash
python gid.py /path/to/your/images
```

This will:
1. Process all images in the specified folder
2. Create a "Described" subfolder
3. Generate a `descriptions.tsv` file with all descriptions
4. Copy processed images to the "Described" subfolder with descriptive filenames

#### Process a Single Image

```bash
python gid.py /path/to/your/image.jpg
```

This will:
1. Process the single image file
2. Output the short description and long description directly to the console
3. No files are created or copied

### Command-Line Options

```
usage: gid.py [-h] [-t TEMPERATURE] [-l LENGTH] [-n] [-k API_KEY] [-w WORKERS] [-v] [-c CONFIG] [-m MODEL] [path]

positional arguments:
  path                  Path to folder or single image file.

options:
  -h, --help            show this help message and exit
  -t TEMPERATURE, --temperature TEMPERATURE
                        Sampling temperature for OpenAI (default=1.0).
  -l LENGTH, --length LENGTH
                        Max tokens (default=4000).
  --init-tsv            Generate TSV with hashes and empty descriptions/context (folder mode only).
  -n, --no-copy         If provided, do NOT copy files to the output folder (folder mode only).
  -k API_KEY, --api-key API_KEY
                        OpenAI API key (overrides config file and environment variable).
  -w WORKERS, --workers WORKERS
                        Maximum number of concurrent workers (folder mode only).
  -v, --verbose         Enable verbose output including HTTP requests.
  -c CONFIG, --config CONFIG
                        Path to the configuration file (default: config.json in the current directory or ~/.config/gid/config.json)
  -m MODEL, --model MODEL
                        OpenAI model to use (default: gpt-5.2)
```

Temperature defaults to **1.0** unless you override it with `--temperature` or the config file.

### Examples

Process images with higher temperature (more creative descriptions):
```bash
python gid.py /path/to/images --temperature 0.9
```

Process images but don't copy them (just create descriptions.tsv):
```bash
python gid.py /path/to/images --no-copy
```

Process images with longer descriptions:
```bash
python gid.py /path/to/images --length 1200
```

Process images with a specific number of concurrent workers:
```bash
python gid.py /path/to/images --workers 4
```

Process a single image with higher temperature:
```bash
python gid.py /path/to/image.jpg --temperature 0.9
```

Enable verbose mode to see detailed API request logs:
```bash
python gid.py /path/to/images --verbose
```

Use a specific config file:
```bash
python gid.py /path/to/images --config /path/to/my-config.json
```

### Configuration File

You can use a JSON configuration file to customize GID's behavior. The config file can be placed in one of these locations:
- `config.json` in the current directory
- `~/.config/gid/config.json` in your home directory
- Any custom path specified with the `--config` flag

The configuration file supports the following settings:

```json
{
  "api": {
    "api_key": "",         // Your OpenAI API key
    "model": "gpt-5.2"     // OpenAI model to use
  },
  "parameters": {
    "temperature": 1.0,    // Sampling temperature
    "max_tokens": 4000     // Maximum response tokens
  },
  "processing": {
    "no_copy": false,      // Whether to skip copying files
    "max_workers": null,   // Max concurrent workers (null = auto)
    "verbose": false       // Enable verbose logging
  },
  "output": {
    "output_folder_name": "Described",  // Name of output folder
    "tsv_filename": "descriptions.tsv"  // Name of TSV file
  },
  "prompt": {
    "system_prompt": "...", // Custom prompt template
    "short_description_max_words": 10   // Max words in short description
  }
}
```

Command-line arguments will override settings in the configuration file.

### Prompt Output Format

The default prompt asks the model to return exactly two lines:

- `SHORT: <short description>`
- `LONG: <long description>`

GID strips these labels when saving to the TSV. If the model returns unlabeled two-line output, GID will still treat the first line as short and the remainder as long.

## Output Format

### TSV File

The script generates a tab-separated values (TSV) file with the following columns:
1. **OriginalFilename**: The original filename of the image
2. **ShortDescription**: A short description suitable for filenames (10 words or less)
3. **LongDescription**: A detailed description of the image content
4. **Context**: Optional per-image facts provided by a user to improve descriptions
5. **Composite**: `yes` or `no` (case-insensitive). If `yes`, this row represents a composite image set.
6. **SHA1**: A SHA-1 hash of the image file for deduplication

### Initialize a TSV for Context

To create a TSV with hashes and empty description fields (so someone can fill in Context), run:

```bash
python gid.py /path/to/images --init-tsv
```

This does not call the API or copy any files. Then add per-image context in the **Context** column and rerun the tool normally. If **ShortDescription** or **LongDescription** is empty, GID will generate descriptions for that row and append the context to the prompt as additional image facts. The **Composite** column defaults to `no` unless you change it.

### Composite Images

To describe multiple related images as a single composite:

1. Add a new row whose **OriginalFilename** is the shared base name (for example, `sina`).
2. Set **Composite** to `yes`.
3. Ensure the actual files are named like `sina_1.jpg`, `sina_2.jpg`, `sina_3.jpg`, etc.

When GID runs, it will find all matching `base_<number>.<ext>` files, send them together in one request, and save the composite description on the composite row. Component rows are skipped during processing. The composite row's **SHA1** is computed from the ordered list of component filenames + hashes, so changing any component triggers a reprocess.

### Described Folder

Unless `--no-copy` is specified, processed images are copied to a "Described" subfolder with filenames based on their short descriptions.

## License

[Creative Commons Attribution-NonCommercial 4.0 International License](LICENSE)

This project is licensed for non-commercial use only. If you'd like to use this project for commercial purposes, please contact the author.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

This tool uses OpenAI's GPT-5.2 model to generate image descriptions.
