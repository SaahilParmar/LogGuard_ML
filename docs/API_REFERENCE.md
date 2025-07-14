# API Reference

## Core Modules

### logguard_ml.core.ml_model

#### AnomalyDetector

**Primary machine learning class for anomaly detection in log data.**

```python
from logguard_ml.core.ml_model import AnomalyDetector

# Initialize with configuration
config = {
    "ml_model": {
        "contamination": 0.05,
        "random_state": 42,
        "max_samples": "auto"
    }
}
detector = AnomalyDetector(config)

# Detect anomalies
results = detector.detect_anomalies(log_dataframe)
```

**Methods:**

- `detect_anomalies(df: pd.DataFrame) -> pd.DataFrame`: Detect anomalies in log data
- `get_feature_importance() -> Dict[str, float]`: Get feature importance scores
- `save_model(filepath: str) -> None`: Save trained model to disk
- `load_model(filepath: str) -> None`: Load model from disk
- `validate_input(df: pd.DataFrame) -> bool`: Validate input data format

**Parameters:**

- `contamination` (float, default=0.05): Expected proportion of anomalies
- `random_state` (int, default=42): Random seed for reproducibility
- `max_samples` (str|int, default="auto"): Number of samples for training

---

### logguard_ml.core.advanced_ml

#### AdvancedAnomalyDetector

**Enhanced ML detector with multiple algorithms and ensemble methods.**

```python
from logguard_ml.core.advanced_ml import AdvancedAnomalyDetector

# Available algorithms
detector = AdvancedAnomalyDetector(algorithm="ensemble")
# Options: "isolation_forest", "one_class_svm", "local_outlier_factor", "ensemble"

results = detector.detect_anomalies(df, config)
```

**Methods:**

- `detect_anomalies(df: pd.DataFrame, config: Dict) -> pd.DataFrame`
- `get_confidence_scores() -> np.ndarray`: Get anomaly confidence scores
- `get_feature_importance() -> Dict[str, float]`: Feature importance analysis
- `save_model(filepath: str) -> None`: Persist trained model
- `load_model(filepath: str) -> None`: Load pre-trained model

**Configuration:**

```yaml
ml_model:
  algorithm: "ensemble"  # isolation_forest, one_class_svm, local_outlier_factor, ensemble
  contamination: 0.05
  n_estimators: 100
  use_pca: true
  n_components: 50
  feature_extraction:
    use_tfidf: true
    max_features: 1000
    ngram_range: [1, 2]
```

---

### logguard_ml.core.log_parser

#### LogParser

**Intelligent log parsing with pattern recognition and performance optimization.**

```python
from logguard_ml.core.log_parser import LogParser

# Initialize with patterns
config = {
    "log_patterns": {
        "apache": r"(?P<ip>\S+) .* (?P<timestamp>\[.*?\]) \"(?P<request>.*?)\" (?P<status>\d+)",
        "nginx": r"(?P<ip>\S+) .* (?P<timestamp>\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})"
    }
}

parser = LogParser(config)
df = parser.parse_file("access.log")
```

**Methods:**

- `parse_file(filepath: str) -> pd.DataFrame`: Parse entire log file
- `parse_lines(lines: List[str]) -> pd.DataFrame`: Parse list of log lines
- `parse_line(line: str) -> Dict`: Parse single log line
- `get_supported_fields() -> List[str]`: Get extractable field names
- `validate_patterns() -> bool`: Validate configured patterns

**Supported Log Formats:**

- Apache/Nginx access logs
- Application logs (with configurable patterns)
- Syslog format
- Custom regex patterns

---

### logguard_ml.core.monitoring

#### LogMonitor

**Real-time log monitoring with streaming anomaly detection.**

```python
from logguard_ml.core.monitoring import LogMonitor

# Monitor a log file
monitor = LogMonitor("app.log", config)
monitor.start_monitoring()

# Stop monitoring
monitor.stop_monitoring()
```

**Methods:**

- `start_monitoring() -> None`: Begin real-time monitoring
- `stop_monitoring() -> None`: Stop monitoring gracefully
- `add_alert_handler(handler: Callable) -> None`: Add custom alert handler
- `get_statistics() -> Dict`: Get monitoring statistics

**Configuration:**

```yaml
monitoring:
  enabled: true
  buffer_size: 1000
  batch_size: 100
  poll_interval: 1.0
  
alerting:
  enabled: true
  threshold: 0.8
  email:
    smtp_server: "smtp.gmail.com"
    smtp_port: 587
    username: "alerts@company.com"
    password: "password"
    recipients: ["admin@company.com"]
  webhook:
    url: "https://hooks.slack.com/services/..."
    timeout: 30
```

---

### logguard_ml.core.performance

#### PerformanceMonitor

**Performance monitoring and optimization utilities.**

```python
from logguard_ml.core.performance import PerformanceMonitor

# Context manager for monitoring
with PerformanceMonitor("operation_name") as monitor:
    # Your code here
    result = some_operation()

# Get statistics
stats = monitor.get_stats()
print(f"Execution time: {stats['execution_time']:.2f}s")
print(f"Memory usage: {stats['peak_memory_mb']:.1f}MB")
```

**Methods:**

- `start() -> None`: Start monitoring
- `stop() -> None`: Stop monitoring
- `get_stats() -> Dict`: Get performance statistics
- `export_report(filepath: str) -> None`: Export detailed report

**Utilities:**

- `optimize_pandas_settings() -> None`: Optimize pandas for performance
- `optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame`: Optimize DataFrame memory
- `profile_function(func: Callable) -> Callable`: Decorator for function profiling

---

## Plugin System

### logguard_ml.plugins

#### PluginManager

**Central management of LogGuard ML plugins.**

```python
from logguard_ml.plugins import PluginManager

manager = PluginManager()

# Load plugins from directory
manager.load_plugins_from_directory("plugins/")

# Get available plugins
plugins = manager.list_plugins()

# Get plugin instance
detector = manager.get_ml_detector("custom_algorithm")
formatter = manager.get_output_format("custom_format")
```

**Plugin Types:**

1. **ML Detector Plugins**: Custom anomaly detection algorithms
2. **Output Format Plugins**: Custom report formats
3. **Log Parser Plugins**: Custom log parsing logic

#### Creating Custom Plugins

**ML Detector Plugin:**

```python
from logguard_ml.plugins import MLDetectorPlugin

class CustomDetector(MLDetectorPlugin):
    @property
    def name(self):
        return "custom_detector"
    
    @property
    def version(self):
        return "1.0.0"
    
    @property
    def description(self):
        return "My custom detector"
    
    def detect_anomalies(self, df, config):
        # Your detection logic
        df['anomaly'] = False
        return df
    
    def get_feature_importance(self):
        return {"feature1": 0.8, "feature2": 0.2}

# Register plugin
from logguard_ml.plugins import register_ml_detector
register_ml_detector(CustomDetector)
```

**Output Format Plugin:**

```python
from logguard_ml.plugins import OutputFormatPlugin

class XMLFormatter(OutputFormatPlugin):
    @property
    def name(self):
        return "xml_format"
    
    @property
    def version(self):
        return "1.0.0"
    
    @property
    def description(self):
        return "XML output format"
    
    @property
    def file_extension(self):
        return "xml"
    
    def generate_output(self, df, output_path, **kwargs):
        # Generate XML output
        xml_content = df.to_xml()
        with open(output_path, 'w') as f:
            f.write(xml_content)
```

---

## Reports

### logguard_ml.reports.report_generator

#### generate_html_report

**Generate comprehensive HTML reports with interactive visualizations.**

```python
from logguard_ml.reports.report_generator import generate_html_report

# Generate report
report_path = generate_html_report(
    df_with_anomalies,
    output_path="report.html",
    config=config,
    include_plots=True,
    include_statistics=True
)
```

**Parameters:**

- `df` (pd.DataFrame): DataFrame with anomaly detection results
- `output_path` (str): Path for output HTML file
- `config` (Dict): Configuration for report generation
- `include_plots` (bool, default=True): Include interactive plots
- `include_statistics` (bool, default=True): Include statistical summary

**Report Sections:**

1. **Executive Summary**: High-level anomaly statistics
2. **Timeline Analysis**: Anomalies over time
3. **Pattern Analysis**: Common patterns and outliers
4. **Detailed Findings**: Individual anomaly details
5. **Recommendations**: Suggested actions

---

## CLI Reference

### logguard Command

**Main command-line interface for LogGuard ML.**

#### analyze

**Analyze log files for anomalies.**

```bash
# Basic analysis
logguard analyze logs/app.log

# Advanced analysis with ML
logguard analyze logs/app.log --ml --algorithm ensemble

# Custom configuration
logguard analyze logs/app.log --config custom.yaml

# Multiple files
logguard analyze logs/*.log --parallel

# Custom output format
logguard analyze logs/app.log --output results.json --format json
```

**Options:**

- `--ml`: Enable machine learning detection
- `--algorithm`: ML algorithm (isolation_forest, ensemble, etc.)
- `--config`: Custom configuration file
- `--output`: Output file path
- `--format`: Output format (html, json, csv)
- `--parallel`: Enable parallel processing
- `--verbose`: Verbose logging

#### monitor

**Real-time log monitoring.**

```bash
# Monitor single file
logguard monitor logs/app.log

# Monitor with alerts
logguard monitor logs/app.log --alerts

# Custom alert configuration
logguard monitor logs/app.log --alerts --config alert_config.yaml

# Monitor directory
logguard monitor logs/ --recursive
```

**Options:**

- `--alerts`: Enable alerting
- `--config`: Configuration file
- `--recursive`: Monitor subdirectories
- `--buffer-size`: Set buffer size
- `--poll-interval`: Set polling interval

#### profile

**Performance profiling and benchmarking.**

```bash
# Profile file processing
logguard profile logs/large_file.log

# Detailed profiling
logguard profile logs/app.log --detailed

# Compare algorithms
logguard profile logs/app.log --compare-algorithms

# Memory profiling
logguard profile logs/app.log --memory
```

**Options:**

- `--detailed`: Detailed profiling information
- `--compare-algorithms`: Compare different ML algorithms
- `--memory`: Include memory profiling
- `--iterations`: Number of profiling iterations

---

## Configuration

### Complete Configuration Schema

```yaml
# Log parsing configuration
log_patterns:
  apache: "(?P<ip>\\S+) .* (?P<timestamp>\\[.*?\\]) \"(?P<request>.*?)\" (?P<status>\\d+)"
  nginx: "(?P<ip>\\S+) .* (?P<timestamp>\\d{4}/\\d{2}/\\d{2} \\d{2}:\\d{2}:\\d{2})"
  custom: "(?P<timestamp>\\S+ \\S+) (?P<level>\\S+) (?P<message>.*)"

# Machine learning configuration
ml_model:
  algorithm: "ensemble"  # isolation_forest, one_class_svm, local_outlier_factor, ensemble
  contamination: 0.05
  random_state: 42
  n_estimators: 100
  
  # Feature extraction
  feature_extraction:
    use_tfidf: true
    max_features: 1000
    ngram_range: [1, 2]
    use_temporal_features: true
    use_statistical_features: true
  
  # Dimensionality reduction
  use_pca: false
  n_components: 50

# Performance settings
performance:
  use_parallel_parsing: true
  chunk_size: 10000
  max_workers: 4
  use_memory_optimization: true
  cache_enabled: true
  cache_size_mb: 100

# Monitoring configuration
monitoring:
  enabled: false
  buffer_size: 1000
  batch_size: 100
  poll_interval: 1.0
  file_patterns: ["*.log"]

# Alerting configuration
alerting:
  enabled: false
  threshold: 0.8
  throttle_minutes: 5
  
  email:
    enabled: false
    smtp_server: "smtp.gmail.com"
    smtp_port: 587
    use_tls: true
    username: "alerts@company.com"
    password: "your_password"
    recipients: ["admin@company.com"]
  
  webhook:
    enabled: false
    url: "https://hooks.slack.com/services/..."
    timeout: 30
    headers:
      Content-Type: "application/json"

# Reporting configuration
reporting:
  include_plots: true
  include_statistics: true
  plot_style: "modern"
  color_scheme: "viridis"
  max_entries_plot: 1000

# Plugin configuration
plugins:
  directories: ["plugins/", "~/.logguard/plugins/"]
  auto_load: true
  
# Logging configuration
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logguard.log"
```

---

## Error Handling

### Custom Exceptions

- `LogParsingError`: Errors during log parsing
- `AnomalyDetectionError`: ML model errors
- `ReportGenerationError`: Report creation errors
- `PluginError`: Plugin loading/management errors
- `ConfigurationError`: Configuration validation errors

### Error Recovery

LogGuard ML implements robust error handling:

1. **Graceful Degradation**: Continues processing when possible
2. **Detailed Logging**: Comprehensive error information
3. **Retry Logic**: Automatic retry for transient failures
4. **User Feedback**: Clear error messages and suggestions

---

## Performance Optimization

### Best Practices

1. **Large Files**: Use parallel processing and chunking
2. **Memory Constraints**: Enable memory optimization
3. **Real-time Processing**: Adjust buffer sizes and batch processing
4. **Multiple Files**: Use parallel file processing

### Tuning Parameters

```yaml
# For large files (100K+ entries)
performance:
  use_parallel_parsing: true
  chunk_size: 50000
  max_workers: 8
  use_memory_optimization: true

# For real-time processing
monitoring:
  buffer_size: 500
  batch_size: 50
  poll_interval: 0.5

# For memory-constrained environments
performance:
  chunk_size: 5000
  max_workers: 2
  use_memory_optimization: true
  
ml_model:
  use_pca: true
  n_components: 25
```

---

## Examples

### Complete Workflow Example

```python
import pandas as pd
from logguard_ml.core.log_parser import LogParser
from logguard_ml.core.advanced_ml import AdvancedAnomalyDetector
from logguard_ml.reports.report_generator import generate_html_report

# 1. Parse logs
config = {
    "log_patterns": {
        "apache": r"(?P<ip>\S+) .* (?P<timestamp>\[.*?\]) \"(?P<request>.*?)\" (?P<status>\d+)"
    }
}

parser = LogParser(config)
df = parser.parse_file("access.log")

# 2. Detect anomalies
detector = AdvancedAnomalyDetector(algorithm="ensemble")
df_with_anomalies = detector.detect_anomalies(df, config)

# 3. Generate report
report_path = generate_html_report(
    df_with_anomalies,
    "anomaly_report.html",
    config,
    include_plots=True
)

print(f"Report generated: {report_path}")
print(f"Found {df_with_anomalies['anomaly'].sum()} anomalies")
```

### Plugin Development Example

```python
# my_plugin.py
from logguard_ml.plugins import MLDetectorPlugin
import pandas as pd
import numpy as np

class StatisticalDetector(MLDetectorPlugin):
    """Statistical anomaly detection using Z-score."""
    
    @property
    def name(self):
        return "statistical_detector"
    
    @property
    def version(self):
        return "1.0.0"
    
    @property
    def description(self):
        return "Z-score based statistical anomaly detection"
    
    def detect_anomalies(self, df, config):
        threshold = config.get('threshold', 2.0)
        
        # Convert message length to z-score
        df['message_length'] = df['message'].str.len()
        z_scores = np.abs((df['message_length'] - df['message_length'].mean()) / df['message_length'].std())
        
        df['anomaly'] = z_scores > threshold
        df['anomaly_score'] = z_scores
        
        return df
    
    def get_feature_importance(self):
        return {"message_length": 1.0}

# Register the plugin
from logguard_ml.plugins import register_ml_detector
register_ml_detector(StatisticalDetector)
```

### Real-time Monitoring Example

```python
from logguard_ml.core.monitoring import LogMonitor
import time

def custom_alert_handler(anomalies):
    """Custom alert handler for anomalies."""
    for anomaly in anomalies:
        print(f"ALERT: {anomaly['message']} (score: {anomaly['anomaly_score']:.2f})")

# Configuration with alerting
config = {
    "monitoring": {
        "enabled": True,
        "buffer_size": 1000,
        "batch_size": 100
    },
    "alerting": {
        "enabled": True,
        "threshold": 0.8
    }
}

# Start monitoring
monitor = LogMonitor("app.log", config)
monitor.add_alert_handler(custom_alert_handler)
monitor.start_monitoring()

try:
    # Monitor for 1 hour
    time.sleep(3600)
finally:
    monitor.stop_monitoring()
```

---

This API reference provides comprehensive documentation for all LogGuard ML components, making it easy for developers to integrate and extend the framework.
