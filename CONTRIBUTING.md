# Contributing to LogGuard ML

We welcome contributions! Here's how you can help make LogGuard ML better.

## Quick Start

### Automated Setup (Recommended)

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YourUsername/LogGuard_ML.git
   cd LogGuard_ML
   ```

2. **Run the setup script**
   ```bash
   ./scripts/setup-dev.sh
   ```

3. **Verify your environment**
   ```bash
   source .venv/bin/activate
   ./scripts/test-env.sh
   ```

### Manual Setup

1. **Fork the repository**
2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install the package in editable mode:**
   ```bash
   pip install -e .
   ```

4. **Install development dependencies:**
   ```bash
   pip install pytest pytest-cov black isort flake8 mypy build twine safety bandit
   ```

5. **Install pre-commit hooks (optional but recommended):**
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Development Workflow

### Running Tests

```bash
# Activate virtual environment
source .venv/bin/activate

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=logguard_ml --cov-report=html

# Run specific test file
pytest tests/test_log_parser.py -v
```

### Code Quality Checks

```bash
# Format code
black logguard_ml tests

# Sort imports
isort logguard_ml tests

# Lint code
flake8 logguard_ml tests

# Type checking
mypy logguard_ml --ignore-missing-imports

# Security scan
bandit -r logguard_ml/
```

### Building and Testing Package

```bash
# Build package
python -m build

# Test package installation
twine check dist/*
```

## Code Style

- Follow PEP 8 guidelines
- Add docstrings to all functions and classes
- Include type hints where appropriate
- Add tests for new features

## Submitting Changes

1. Create a feature branch: `git checkout -b feature-name`
2. Make your changes and add tests
3. Ensure all tests pass
4. Submit a pull request with a clear description

## Reporting Issues

Please use the GitHub issue tracker to report bugs or suggest features.
