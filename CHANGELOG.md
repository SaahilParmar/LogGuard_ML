# Changelog

All notable changes to LogGuard ML will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Professional package structure with proper Python packaging
- Comprehensive command-line interface (CLI) with `logguard` command
- Enhanced error handling with custom exception classes
- Type hints throughout the codebase
- Comprehensive test suite with pytest
- Code quality tools integration (Black, isort, flake8, mypy)
- Pre-commit hooks for automated code quality checks
- GitHub Actions CI/CD pipeline
- Professional HTML reports with interactive visualizations
- Detailed logging with configurable levels
- Support for multiple output formats (HTML, JSON, CSV)
- Enhanced anomaly detection with TF-IDF text features
- Improved configuration validation
- Development dependencies and tools setup

### Changed
- Restructured project as proper Python package (`logguard_ml`)
- Enhanced log parser with better error handling and logging
- Improved ML model with feature engineering and validation
- Modern report generator with Plotly visualizations
- Updated requirements with proper version pinning
- Enhanced README with professional documentation

### Deprecated
- Legacy `main.py` interface (still functional but deprecated)

### Fixed
- Better handling of malformed log entries
- Improved memory efficiency for large log files
- Fixed edge cases in anomaly detection
- Enhanced configuration file validation

## [0.1.0] - 2025-01-15

### Added
- Initial release of LogGuard ML
- Basic log parsing with regex patterns
- Isolation Forest anomaly detection
- HTML report generation
- YAML configuration support
- Command-line interface
- Sample log files and configuration
- Basic test suite
- MIT License
- README documentation

### Features
- Intelligent log parsing with configurable patterns
- Machine learning-based anomaly detection
- Beautiful HTML reports with visualizations
- Fast processing of large log files
- Configurable through YAML files
