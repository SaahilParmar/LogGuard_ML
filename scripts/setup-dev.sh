#!/bin/bash
# Development Environment Setup Script for LogGuard ML

set -e  # Exit on any error

echo "🚀 Setting up LogGuard ML development environment..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed. Please install Python 3.9 or higher."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.9"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
    echo "❌ Python $REQUIRED_VERSION or higher is required. Found: $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python $PYTHON_VERSION detected"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install package in editable mode
echo "📥 Installing LogGuard ML in editable mode..."
pip install -e .

# Install development dependencies
echo "🛠️  Installing development dependencies..."
pip install pytest pytest-cov black isort flake8 mypy build twine safety bandit

# Install pre-commit hooks if config exists
if [ -f ".pre-commit-config.yaml" ]; then
    echo "🪝 Installing pre-commit hooks..."
    pip install pre-commit
    pre-commit install
fi

echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "    source .venv/bin/activate"
echo ""
echo "To run tests:"
echo "    pytest tests/ -v"
echo ""
echo "To run code quality checks:"
echo "    black logguard_ml tests"
echo "    isort logguard_ml tests"
echo "    flake8 logguard_ml tests"
echo "    mypy logguard_ml"
echo ""
echo "To deactivate the environment:"
echo "    deactivate"
