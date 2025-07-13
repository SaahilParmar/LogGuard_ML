#!/bin/bash
# Cleanup script to remove build artifacts and cache files

echo "🧹 Cleaning up build artifacts and cache files..."

# Remove Python cache files
echo "🗑️  Removing Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true

# Remove build artifacts
echo "🗑️  Removing build artifacts..."
rm -rf build/
rm -rf dist/
rm -rf *.egg-info/
rm -rf .eggs/

# Remove test and coverage artifacts
echo "🗑️  Removing test artifacts..."
rm -rf .pytest_cache/
rm -rf .coverage
rm -rf htmlcov/
rm -rf coverage.xml
rm -rf .tox/

# Remove other temporary files
echo "🗑️  Removing temporary files..."
rm -rf .cache/
rm -rf .mypy_cache/
rm -rf .ruff_cache/

echo "✅ Cleanup complete!"
echo ""
echo "Note: This script only removes build artifacts and cache files."
echo "Source code and configuration files are preserved."
