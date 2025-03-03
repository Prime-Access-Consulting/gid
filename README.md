# GID - Generate Image Descriptions

GID is a Python tool for automatically generating human-readable descriptions of images using OpenAI's GPT-4o Vision model. It processes image files in a specified folder, creates both short and detailed descriptions, and optionally renames and copies the files based on their descriptions.

## Features

- **Dual Descriptions**: Generates both short (filename-friendly) and detailed descriptions
- **Efficient Processing**: Processes images in parallel using multiple threads
- **Deduplication**: Uses SHA-1 hashing to avoid reprocessing duplicate images
- **Organized Output**: 
  - Stores all descriptions in a TSV file for easy reference
  - Optionally copies images to a "Described" subfolder with descriptive filenames
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

3. Set your OpenAI API key:
   ```bash
   # Linux/macOS
   export openai_api=your_api_key_here
   
   # Windows (Command Prompt)
   set openai_api=your_api_key_here
   
   # Windows (PowerShell)
   $env:openai_api="your_api_key_here"
   ```

## Usage

### Basic Usage

```bash
python gid.py /path/to/your/images
```

This will:
1. Process all images in the specified folder
2. Create a "Described" subfolder
3. Generate a `descriptions.tsv` file with all descriptions
4. Copy processed images to the "Described" subfolder with descriptive filenames

### Command-Line Options

```
usage: gid.py [-h] [-t TEMPERATURE] [-l LENGTH] [-n] [-k API_KEY] [-w WORKERS] [-v] [folder]

positional arguments:
  folder                Path to folder containing images.

options:
  -h, --help            show this help message and exit
  -t TEMPERATURE, --temperature TEMPERATURE
                        Sampling temperature for OpenAI (default=0.7).
  -l LENGTH, --length LENGTH
                        Max tokens (default=800).
  -n, --no-copy         If provided, do NOT copy files to 'Described' folder.
  -k API_KEY, --api-key API_KEY
                        OpenAI API key (overrides environment variable).
  -w WORKERS, --workers WORKERS
                        Maximum number of concurrent workers (default is auto).
  -v, --verbose         Enable verbose output including HTTP requests.
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

Enable verbose mode to see detailed API request logs:
```bash
python gid.py /path/to/images --verbose
```

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