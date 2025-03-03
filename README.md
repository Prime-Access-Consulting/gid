# GID - Generate Image Descriptions

GID is a Python tool for automatically generating human-readable descriptions of images using OpenAI's GPT-4o Vision model. It processes image files in a specified folder, creates both short and detailed descriptions, and optionally renames and copies the files based on their descriptions.

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

## Installation

### Prerequisites

- Python 3.7+
- OpenAI API key with access to GPT-4o Vision model

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
   export openai_api=your_api_key_here
   
   # Windows (Command Prompt)
   set openai_api=your_api_key_here
   
   # Windows (PowerShell)
   $env:openai_api="your_api_key_here"
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
                        Sampling temperature for OpenAI (default=0.7).
  -l LENGTH, --length LENGTH
                        Max tokens (default=800).
  -n, --no-copy         If provided, do NOT copy files to the output folder (folder mode only).
  -k API_KEY, --api-key API_KEY
                        OpenAI API key (overrides config file and environment variable).
  -w WORKERS, --workers WORKERS
                        Maximum number of concurrent workers (folder mode only).
  -v, --verbose         Enable verbose output including HTTP requests.
  -c CONFIG, --config CONFIG
                        Path to the configuration file (default: config.json in the current directory or ~/.config/gid/config.json)
  -m MODEL, --model MODEL
                        OpenAI model to use (default: gpt-4o)
```

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
    "model": "gpt-4o"      // OpenAI model to use
  },
  "parameters": {
    "temperature": 0.7,    // Sampling temperature
    "max_tokens": 800      // Maximum response tokens
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

## Output Format

### TSV File

The script generates a tab-separated values (TSV) file with the following columns:
1. **OriginalFilename**: The original filename of the image
2. **ShortDescription**: A short description suitable for filenames (10 words or less)
3. **LongDescription**: A detailed description of the image content
4. **SHA1**: A SHA-1 hash of the image file for deduplication

### Described Folder

Unless `--no-copy` is specified, processed images are copied to a "Described" subfolder with filenames based on their short descriptions.

## License

[Creative Commons Attribution-NonCommercial 4.0 International License](LICENSE)

This project is licensed for non-commercial use only. If you'd like to use this project for commercial purposes, please contact the author.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

This tool uses OpenAI's GPT-4o Vision model to generate image descriptions.