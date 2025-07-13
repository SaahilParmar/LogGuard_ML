# Contributing to LogGuard ML

We welcome contributions! Here's how you can help:

## Development Setup

1. Fork the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running Tests

```bash
# Run all tests
PYTHONPATH=. pytest

# Run with coverage
PYTHONPATH=. pytest --cov=utils --cov-report=html
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
