#!/bin/bash
# Quick test script to verify the development environment

set -e

echo "🧪 Running LogGuard ML environment tests..."

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "❌ Virtual environment not activated. Please run:"
    echo "    source .venv/bin/activate"
    exit 1
fi

echo "✅ Virtual environment active: $VIRTUAL_ENV"

# Check if package is installed
if ! python -c "import logguard_ml" 2>/dev/null; then
    echo "❌ LogGuard ML package not installed. Please run:"
    echo "    pip install -e ."
    exit 1
fi

echo "✅ LogGuard ML package is installed"

# Run quick import test
echo "🔍 Testing package imports..."
python -c "
from logguard_ml.core.log_parser import LogParser
from logguard_ml.core.ml_model import AnomalyDetector
from logguard_ml.reports.report_generator import generate_html_report
from logguard_ml.cli import main
print('✅ All core modules imported successfully')
"

# Check if CLI is available
if ! command -v logguard &> /dev/null; then
    echo "⚠️  CLI command 'logguard' not found in PATH. This is normal if package is not installed globally."
else
    echo "✅ CLI command 'logguard' is available"
fi

# Run a quick test
echo "🏃 Running quick tests..."
python -m pytest tests/ -x -q --tb=short

echo ""
echo "🎉 Environment verification complete!"
echo "Your development environment is ready to use."
