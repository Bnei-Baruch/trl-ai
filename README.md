# TRL AI Project

This is a Python project using Conda for environment management.

## Setup

1. Install Conda (if not already installed)
   - Download and install Miniconda from: https://docs.conda.io/en/latest/miniconda.html
   - Or install Anaconda from: https://www.anaconda.com/products/distribution

2. Create the conda environment:
   ```bash
   conda env create -f environment.yml
   ```

3. Activate the environment:
   ```bash
   conda activate trlai
   ```

## Project Structure

```
trlai/
├── environment.yml    # Conda environment configuration
├── README.md         # Project documentation
├── src/             # Source code directory
├── tests/           # Test files
└── notebooks/       # Jupyter notebooks
```

## Development

- Use `black` for code formatting
- Use `flake8` for linting
- Run tests with `pytest`

## License

MIT 