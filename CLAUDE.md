# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

scvi-tools (single-cell variational inference tools) is a Python package for probabilistic modeling and analysis of single-cell omics data, built on PyTorch and AnnData.

## Development Commands

### Installation
```bash
pip install -e ".[dev]"
# or with uv
uv pip install -e ".[dev]"
```

### Linting & Formatting
The project uses **Ruff** for linting and formatting:
```bash
ruff check .          # check for linting errors
ruff check --fix .    # auto-fix linting errors
ruff format .         # format code
```

Pre-commit hooks are available:
```bash
pre-commit install    # install hooks
pre-commit run --all  # run all checks
```

### Testing
Tests are located in `tests/` directory:
```bash
pytest                              # run all tests
pytest tests/model                  # run tests in a specific directory
pytest tests/model/test_scvi.py     # run tests in a specific file
pytest tests/model/test_scvi.py::test_function  # run a specific test
```

Test markers (defined in pyproject.toml):
- `internet` - requires internet access
- `optional` - optional tests that take more time
- `private` - uses private keys (e.g., HuggingFace)
- `multigpu` - multi-GPU tests
- `autotune` - Ray autotune tests
- `jax` - JAX-related tests

## Code Architecture

### Package Structure (`src/scvi/`)

- **`model/`** - High-level model API (SCVI, SCANVI, TOTALVI, MULTIVI, PeakVI, CondSCVI, DestVI, AutoZI, etc.)
- **`module/`** - Core neural network modules (VAE, etc.)
- **`nn/`** - Neural network building blocks and architectures
- **`train/`** - Training plans and data splitters
- **`dataloaders/`** - Data loading utilities
- **`data/`** - Data handling, fields, and built-in datasets
- **`distributions/`** - Custom probability distributions
- **`external/`** - External model implementations (GIMVI, Tangram, etc.)
- **`hub/`** - Hugging Face Hub integration
- **`autotune/`** - Hyperparameter tuning with Ray
- **`criticism/`** - Posterior predictive checks
- **`utils/`** - Utility functions

### Key Patterns

1. **Model Architecture**: Models follow a layered architecture:
   - `model/` - User-facing API
   - `module/` - Core computational logic (VAE, etc.)
   - `nn/` - Neural network components

2. **Registration Pattern**: Models register with AnnData via `setup_anndata()` class method

3. **Training**: Uses PyTorch Lightning via custom training plans in `train/`

4. **Data Handling**: Data is managed through `AnnData`/`MuData` objects with specialized field types in `data/fields/`

## Code Style

- Python 3.12+ required
- Line length: 99 characters
- Docstrings: NumPy convention
- Uses Ruff for formatting (double quotes, space indentation)

## Testing Patterns

- Tests mirror the `src/scvi/` structure in `tests/`
- Use pytest fixtures from `tests/conftest.py`
- Mock data utilities available in test files
