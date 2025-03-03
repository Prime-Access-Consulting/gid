# GID - Generate Image Descriptions

## Build/Run Commands
```bash
# Run the script (replace folder_path with target directory)
python gid.py /path/to/images

# Run with custom parameters
python gid.py /path/to/images --temperature 0.8 --length 1000 --no-copy
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
- Generates TSV with columns: OriginalFilename, ShortDescription, LongDescription, SHA1
- Uses OpenAI API (GPT-4o) for image description generation