# LogGuard ML

🛡️ **AI-Powered Log Analysis & Anomaly Detection Framework**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)](./tests/)

LogGuard ML combines regex-based log parsing with machine learning to automatically detect anomalies in application logs and generate beautiful, actionable reports.

---

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/SaahilParmar/LogGuard_ML.git
cd LogGuard_ML

# Install dependencies
pip install -r requirements.txt

# Run basic log analysis
python main.py --logfile data/sample_log.log

# Run with ML anomaly detection
python main.py --logfile data/sample_log.log --ml
```

**Output**: Generates `reports/anomaly_report.html` with beautiful visualizations

---

## ✨ Features

- 🔍 **Intelligent Log Parsing** - Configurable regex patterns via YAML
- 🤖 **ML Anomaly Detection** - Isolation Forest algorithm for outlier detection  
- 📊 **Beautiful Reports** - Professional HTML reports with modern styling
- ⚡ **Fast & Lightweight** - Processes thousands of log entries in seconds
- 🧪 **Fully Tested** - Comprehensive unit tests with pytest
- 🔧 **Configurable** - Easily adapt to different log formats

---

## 📂 Project Structure

```
LogGuard_ML/
├── 📁 config/              
│   └── config.yaml         # Log patterns & ML configuration
├── 📁 data/                
│   └── sample_log.log      # Sample data for testing
├── 📁 reports/             
│   ├── __init__.py         
│   └── report_generator.py # HTML report generation
├── 📁 utils/               
│   ├── log_parser.py       # Core parsing logic
│   └── ml_model.py         # Anomaly detection ML
├── 📁 tests/               
│   └── test_log_parser.py  # Unit tests
├── main.py                 # CLI entry point
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

---

## 🧪 Testing

```bash
# Run all tests with proper path
PYTHONPATH=. pytest

# Run specific test file  
PYTHONPATH=. pytest tests/test_log_parser.py

# Run with coverage report
PYTHONPATH=. pytest --cov=utils --cov-report=html
```

---

## 🔧 Configuration

Customize log parsing and ML behavior by editing `config/config.yaml`:

```yaml
# Define regex patterns for your log format
log_patterns:
  - pattern: "(?P<timestamp>\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}) (?P<level>ERROR|WARN|INFO) (?P<message>.+)"

# Configure ML anomaly detection
ml_model:
  algorithm: isolation_forest
  contamination: 0.05        # % of data considered anomalous
  random_state: 42
```

---

## 📊 Usage Examples

### Basic Log Analysis
```bash
python main.py --logfile data/sample_log.log
```
*Parses logs and generates basic HTML report*

### ML-Enhanced Anomaly Detection  
```bash
python main.py --logfile data/sample_log.log --ml
```
*Adds machine learning to detect suspicious patterns*

### Custom Configuration
```bash
python main.py --logfile logs/app.log --config custom_config.yaml --output reports/custom_report.html
```
*Use custom settings and output location*

---

## 🎯 How It Works

1. **Parse**: Extracts structured data from raw log files using regex
2. **Analyze**: Applies ML algorithms to identify anomalous patterns  
3. **Report**: Generates interactive HTML reports with findings
4. **Alert**: Highlights suspicious entries for further investigation

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup instructions
- Code style guidelines  
- Testing requirements
- Pull request process

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built with [scikit-learn](https://scikit-learn.org/) for machine learning
- Styled reports using modern CSS and responsive design
- Tested with [pytest](https://pytest.org/) framework
