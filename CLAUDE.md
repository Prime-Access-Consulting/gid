# GID - Generate Image Descriptions

## Build/Run Commands
```bash
# Process a folder of images (creates TSV and optional copies)
python gid.py /path/to/images

# Process a single image (outputs descriptions to console only)
python gid.py /path/to/image.jpg

# Run with custom parameters
python gid.py /path/to/images --temperature 0.8 --length 1000 --no-copy

# Use a config file
python gid.py /path/to/images --config /path/to/config.json

# Specify a different OpenAI model
python gid.py /path/to/images --model gpt-4-vision-preview
```

## Code Style Guidelines
- **Imports**: Standard library first, then third-party, then local modules; alphabetized within groups
- **Formatting**: 4-space indentation, 88 character line length
- **Docstrings**: Triple quotes for all functions/classes with concise description
- **Naming**: snake_case for variables/functions, CamelCase for classes
- **Error Handling**: Use try/except with specific exceptions and meaningful error messages
- **Concurrency**: Use ThreadPoolExecutor for parallelizable operations
- **Type Hints**: Not currently used, but consider adding in future updates

## Project Structure
- Single Python script with modular functions
- Two operation modes:
  - Folder mode: Generates TSV with columns: OriginalFilename, ShortDescription, LongDescription, SHA1
  - Single image mode: Outputs short and long descriptions directly to console
- Uses OpenAI API (GPT-4o) for image description generation