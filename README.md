# LogGuard ML

🛡️ **AI-Powered Log Analysis & Anomaly Detection Framework**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)](./tests/)
[![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen.svg)](./tests/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

LogGuard ML is a professional-grade framework that combines intelligent log parsing with machine learning to automatically detect anomalies in application logs. It generates beautiful, actionable HTML reports with interactive visualizations and provides a modern CLI for seamless integration into your monitoring workflows.

---

## ✨ Key Features

- 🔍 **Intelligent Log Parsing** - Configurable regex patterns via YAML with robust error handling
- 🤖 **Advanced ML Anomaly Detection** - Isolation Forest with TF-IDF text features and temporal analysis
- 📊 **Interactive Reports** - Professional HTML reports with Plotly visualizations and Bootstrap styling
- ⚡ **High Performance** - Optimized for processing thousands of log entries with memory efficiency
- 🛠️ **Professional CLI** - Modern command-line interface with comprehensive options
- 🧪 **Fully Tested** - Comprehensive test suite with 90%+ coverage
- 🔧 **Highly Configurable** - Flexible configuration system for different log formats
- 📦 **Easy Installation** - Pip-installable package with proper dependency management

---

## 🚀 Quick Start

### Installation

```bash
# Install from source (recommended for now)
git clone https://github.com/SaahilParmar/LogGuard_ML.git
cd LogGuard_ML
pip install -e .

# Or install development version
pip install -e ".[dev]"
```

### Basic Usage

```bash
# Analyze logs with ML anomaly detection
logguard analyze data/sample_log.log --ml

# Generate different output formats
logguard analyze app.log --ml --format json --output results.json
logguard analyze app.log --format csv --output results.csv

# Custom configuration and verbose output
logguard analyze app.log --config custom_config.yaml --ml --verbose

# Get help
logguard --help
logguard analyze --help
```

### Python API

```python
from logguard_ml import LogParser, AnomalyDetector, generate_html_report

# Load configuration
import yaml
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Parse logs
parser = LogParser(config)
df = parser.parse_log_file("application.log")

# Detect anomalies
detector = AnomalyDetector(config)
df_with_anomalies = detector.detect_anomalies(df)

# Generate report
generate_html_report(df_with_anomalies, "anomaly_report.html")
```

---

## 📂 Project Structure

```
LogGuard_ML/
├── 📦 logguard_ml/              # Main package
│   ├── 🧠 core/                 # Core functionality  
│   │   ├── log_parser.py        # Intelligent log parsing
│   │   └── ml_model.py          # ML anomaly detection
│   ├── 📊 reports/              # Report generation
│   │   └── report_generator.py  # HTML report creation
│   ├── ⚙️ config/              # Configuration files
│   │   └── config.yaml          # Default configuration
│   ├── 🖥️ cli.py               # Command-line interface
│   └── 📄 __init__.py          # Package initialization
├── 🧪 tests/                   # Comprehensive test suite
├── 📁 data/                    # Sample data files
├── 📄 docs/                    # Documentation
└── 🔧 Development files        # Setup, CI/CD, etc.
```

---

## ⚙️ Configuration

LogGuard ML uses YAML configuration files for maximum flexibility:

```yaml
# Log parsing patterns
log_patterns:
  - pattern: "(?P<timestamp>\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}) (?P<level>ERROR|WARN|INFO) (?P<message>.+)"

# ML model parameters  
ml_model:
  algorithm: isolation_forest
  contamination: 0.05        # Expected anomaly percentage
  random_state: 42
  max_samples: auto

# Report settings
report:
  output_html: reports/anomaly_report.html
  include_raw_data: true
  max_anomalies_display: 100
```

### Supported Log Formats

LogGuard ML can parse any log format by configuring regex patterns:

- **Apache/Nginx** access logs
- **Application** logs (Java, Python, Node.js)
- **System** logs (syslog, Windows Event Log)
- **Custom** formats via regex patterns

---

## 🤖 Machine Learning Features

### Anomaly Detection Algorithm
- **Isolation Forest** - Efficient unsupervised anomaly detection
- **Feature Engineering** - Message length, severity scores, temporal patterns
- **Text Analysis** - TF-IDF vectorization for message content analysis
- **Configurable Sensitivity** - Adjustable contamination parameters

### Features Extracted
- Message length and complexity
- Log level severity scores  
- Temporal patterns (hour, day of week)
- Text content similarity (TF-IDF)
- Custom regex-based features

---

## 📊 Report Features

### Interactive Visualizations
- **Log Level Distribution** - Pie charts showing log severity breakdown
- **Anomaly Timeline** - Time-series view of anomalies vs normal logs
- **Message Length Analysis** - Histograms comparing normal vs anomalous entries
- **Summary Statistics** - Key metrics and insights

### Professional Styling
- **Bootstrap 5** - Modern, responsive design
- **Plotly Charts** - Interactive, zoomable visualizations
- **Professional Layout** - Clean, organized presentation
- **Export Options** - Print-friendly and shareable formats

---

## 🧪 Development & Testing

### Development Setup
```bash
# Clone repository
git clone https://github.com/SaahilParmar/LogGuard_ML.git
cd LogGuard_ML

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=logguard_ml --cov-report=html

# Run specific test categories
pytest tests/test_comprehensive.py -v
```

### Code Quality
```bash
# Format code
black logguard_ml tests

# Sort imports  
isort logguard_ml tests

# Lint code
flake8 logguard_ml tests

# Type checking
mypy logguard_ml

# Run all quality checks
pre-commit run --all-files
```

---

## 🚀 Advanced Usage

### Batch Processing
```bash
# Process multiple log files
for log in logs/*.log; do
    logguard analyze "$log" --ml --output "reports/$(basename $log .log)_report.html"
done
```

### Custom Configuration
```bash
# Use custom patterns for specific log formats
logguard analyze app.log --config configs/java_app_config.yaml --ml
```

### Integration with Monitoring
```bash
# Generate JSON output for monitoring systems
logguard analyze app.log --ml --format json | jq '.anomalies[] | select(.anomaly_score < -0.5)'
```

---

## 🛠️ API Reference

### LogParser Class
```python
class LogParser:
    def __init__(self, config: Dict) -> None: ...
    def parse_log_file(self, filepath: Union[str, Path]) -> pd.DataFrame: ...
    def get_supported_fields(self) -> List[str]: ...
```

### AnomalyDetector Class  
```python
class AnomalyDetector:
    def __init__(self, config: Dict) -> None: ...
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame: ...
    def save_model(self, filepath: str) -> None: ...
    def load_model(self, filepath: str) -> None: ...
```

### Report Generation
```python
def generate_html_report(
    df: pd.DataFrame,
    output_path: str = "reports/anomaly_report.html", 
    title: str = "LogGuard ML - Anomaly Detection Report",
    include_raw_data: bool = True
) -> None: ...
```

---

## 🔄 CI/CD & Automation

LogGuard ML includes a complete CI/CD pipeline:

- **Automated Testing** - Multi-Python version testing (3.9-3.12)
- **Code Quality Checks** - Linting, formatting, type checking
- **Security Scanning** - Dependency vulnerability checks
- **Documentation** - Automated API documentation generation
- **Package Building** - Automated PyPI package creation

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Quick Contribution Steps
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run quality checks (`pre-commit run --all-files`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)  
7. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **scikit-learn** - Machine learning algorithms
- **Plotly** - Interactive visualizations  
- **pandas** - Data manipulation and analysis
- **Bootstrap** - Professional UI styling
- **pytest** - Comprehensive testing framework

---

## 📞 Support & Community

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/SaahilParmar/LogGuard_ML/issues)
- 💡 **Feature Requests**: [GitHub Issues](https://github.com/SaahilParmar/LogGuard_ML/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/SaahilParmar/LogGuard_ML/discussions)
- 📖 **Documentation**: [GitHub Wiki](https://github.com/SaahilParmar/LogGuard_ML/wiki)

---

<div align="center">

**⭐ Star this repository if LogGuard ML helps you!**

Made with ❤️ by the LogGuard ML community

</div>

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
