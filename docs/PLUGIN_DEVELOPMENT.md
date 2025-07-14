# Plugin Development Guide

## Overview

LogGuard ML provides a powerful plugin architecture that allows you to extend the system's capabilities without modifying the core codebase. This guide covers everything you need to know about developing plugins for LogGuard ML.

## Plugin Types

LogGuard ML supports three main types of plugins:

### 1. ML Detector Plugins
Extend the anomaly detection capabilities with custom machine learning algorithms.

**Base Class**: `MLDetectorPlugin`

**Key Methods**:
- `detect_anomalies(df)`: Main detection logic
- `train(df)`: Optional training method
- `get_feature_importance()`: Feature importance analysis

### 2. Output Format Plugins
Create custom output formats for reports and visualizations.

**Base Class**: `OutputFormatPlugin`

**Key Methods**:
- `generate_output(df, output_path, **kwargs)`: Generate formatted output
- `validate_output_path(path)`: Validate output file path

### 3. Log Parser Plugins
Parse custom log formats or add preprocessing capabilities.

**Base Class**: `LogParserPlugin`

**Key Methods**:
- `parse_log_line(line)`: Parse individual log lines
- `parse_log_file(file_path)`: Parse entire log files
- `validate_log_format(content)`: Validate log format

## Plugin Development Process

### Step 1: Choose Plugin Type

Determine which type of plugin you need based on your requirements:

- **ML Detector**: For custom anomaly detection algorithms
- **Output Format**: For custom report formats or visualizations
- **Log Parser**: For custom log formats or preprocessing

### Step 2: Create Plugin File

Create a new Python file in the `plugins/` directory:

```bash
plugins/
├── your_plugin_name.py
└── __init__.py
```

### Step 3: Implement Base Class

Import and inherit from the appropriate base class:

```python
from logguard_ml.plugins import MLDetectorPlugin

class YourDetector(MLDetectorPlugin):
    @property
    def name(self) -> str:
        return "your_detector"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Your detector description"
    
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        # Your implementation here
        pass
```

### Step 4: Register Plugin

Add your plugin to the configuration file (`config/config.yaml`):

```yaml
plugins:
  ml_detectors:
    - name: "your_detector"
      module: "plugins.your_plugin_name"
      class: "YourDetector"
      enabled: true
```

### Step 5: Test Plugin

Create unit tests for your plugin in the `tests/` directory:

```python
import unittest
from plugins.your_plugin_name import YourDetector

class TestYourDetector(unittest.TestCase):
    def test_detection(self):
        detector = YourDetector()
        # Add your test cases
```

## Example: Custom ML Detector

Here's a complete example of a custom ML detector plugin:

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from logguard_ml.plugins import MLDetectorPlugin

class IsolationForestDetector(MLDetectorPlugin):
    """
    Anomaly detection using Isolation Forest algorithm.
    """
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.contamination = 0.1
    
    @property
    def name(self) -> str:
        return "isolation_forest"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Isolation Forest based anomaly detection"
    
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies using Isolation Forest."""
        # Prepare features
        features = self._extract_features(df)
        
        # Train model if not already trained
        if self.model is None:
            self.train(df)
        
        # Predict anomalies
        predictions = self.model.predict(features)
        scores = self.model.decision_function(features)
        
        # Add results to dataframe
        result_df = df.copy()
        result_df['anomaly'] = predictions == -1
        result_df['anomaly_score'] = scores
        
        return result_df
    
    def train(self, df: pd.DataFrame) -> None:
        """Train the Isolation Forest model."""
        features = self._extract_features(df)
        
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=42
        )
        self.model.fit(features)
    
    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract numerical features for the model."""
        features = []
        
        # Message length
        features.append(df['message'].str.len().values)
        
        # Log level encoding
        level_encoding = {'DEBUG': 0, 'INFO': 1, 'WARNING': 2, 'ERROR': 3}
        features.append(df['level'].map(level_encoding).fillna(1).values)
        
        # Hour of day (if timestamp available)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            features.append(df['timestamp'].dt.hour.values)
        
        return np.column_stack(features)
    
    def get_feature_importance(self) -> dict:
        """Return feature importance information."""
        return {
            'message_length': 0.4,
            'log_level': 0.3,
            'hour_of_day': 0.3
        }
```

## Example: Custom Output Format

Here's an example of a custom output format plugin:

```python
import json
from datetime import datetime
from typing import Dict, Any
import pandas as pd
from logguard_ml.plugins import OutputFormatPlugin

class SlackFormat(OutputFormatPlugin):
    """
    Output format for Slack notifications.
    """
    
    @property
    def name(self) -> str:
        return "slack_format"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Slack notification format"
    
    @property
    def file_extension(self) -> str:
        return "json"
    
    def generate_output(self, df: pd.DataFrame, output_path: str, **kwargs) -> None:
        """Generate Slack-formatted notification."""
        # Calculate metrics
        total_logs = len(df)
        anomalies = df[df['anomaly'] == True] if 'anomaly' in df.columns else pd.DataFrame()
        anomaly_count = len(anomalies)
        anomaly_rate = (anomaly_count / total_logs * 100) if total_logs > 0 else 0
        
        # Create Slack message
        slack_message = {
            "text": "LogGuard ML Analysis Report",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "🔍 LogGuard ML Analysis Report"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Total Logs:*\n{total_logs:,}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Anomalies:*\n{anomaly_count:,}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Anomaly Rate:*\n{anomaly_rate:.2f}%"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Generated:*\n{datetime.now().strftime('%Y-%m-%d %H:%M')}"
                        }
                    ]
                }
            ]
        }
        
        # Add severity indicator
        if anomaly_rate > 10:
            slack_message["blocks"].append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "🚨 *HIGH ANOMALY RATE DETECTED* - Immediate investigation recommended"
                }
            })
        elif anomaly_rate > 5:
            slack_message["blocks"].append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "⚠️ *ELEVATED ANOMALY RATE* - Monitor closely"
                }
            })
        
        # Save Slack message
        with open(output_path, 'w') as f:
            json.dump(slack_message, f, indent=2)
```

## Example: Custom Log Parser

Here's an example of a custom log parser plugin:

```python
import re
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional
from logguard_ml.plugins import LogParserPlugin

class NginxLogParser(LogParserPlugin):
    """
    Parser for Nginx access logs.
    """
    
    def __init__(self):
        super().__init__()
        # Nginx combined log format regex
        self.log_pattern = re.compile(
            r'(?P<ip>\S+) \S+ \S+ \[(?P<timestamp>[^\]]+)\] '
            r'"(?P<method>\S+) (?P<url>\S+) (?P<protocol>\S+)" '
            r'(?P<status>\d+) (?P<size>\d+|-) '
            r'"(?P<referrer>[^"]*)" "(?P<user_agent>[^"]*)"'
        )
    
    @property
    def name(self) -> str:
        return "nginx_parser"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Nginx access log parser"
    
    @property
    def supported_formats(self) -> list:
        return ["nginx", "apache"]
    
    def parse_log_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a single Nginx log line."""
        match = self.log_pattern.match(line.strip())
        if not match:
            return None
        
        data = match.groupdict()
        
        # Parse timestamp
        try:
            timestamp = datetime.strptime(
                data['timestamp'], 
                '%d/%b/%Y:%H:%M:%S %z'
            )
        except ValueError:
            timestamp = datetime.now()
        
        # Determine log level based on status code
        status = int(data['status'])
        if status >= 500:
            level = 'ERROR'
        elif status >= 400:
            level = 'WARNING'
        else:
            level = 'INFO'
        
        return {
            'timestamp': timestamp,
            'level': level,
            'message': f"{data['method']} {data['url']} - {status}",
            'ip_address': data['ip'],
            'http_method': data['method'],
            'url': data['url'],
            'status_code': status,
            'response_size': int(data['size']) if data['size'] != '-' else 0,
            'user_agent': data['user_agent'],
            'referrer': data['referrer']
        }
    
    def parse_log_file(self, file_path: str) -> pd.DataFrame:
        """Parse an entire Nginx log file."""
        records = []
        
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                parsed = self.parse_log_line(line)
                if parsed:
                    records.append(parsed)
                else:
                    print(f"Warning: Could not parse line {line_num}")
        
        return pd.DataFrame(records)
    
    def validate_log_format(self, content: str) -> bool:
        """Validate if content matches Nginx log format."""
        lines = content.strip().split('\n')
        if not lines:
            return False
        
        # Check if at least 50% of lines match the pattern
        matches = 0
        total_lines = min(len(lines), 100)  # Check first 100 lines
        
        for line in lines[:total_lines]:
            if self.log_pattern.match(line.strip()):
                matches += 1
        
        return (matches / total_lines) >= 0.5
```

## Advanced Plugin Features

### Configuration

Plugins can accept configuration parameters:

```python
class ConfigurableDetector(MLDetectorPlugin):
    def __init__(self, threshold=0.5, algorithm='isolation_forest'):
        super().__init__()
        self.threshold = threshold
        self.algorithm = algorithm
    
    @classmethod
    def from_config(cls, config: dict):
        """Create instance from configuration."""
        return cls(
            threshold=config.get('threshold', 0.5),
            algorithm=config.get('algorithm', 'isolation_forest')
        )
```

### Plugin Dependencies

Declare optional dependencies in your plugin:

```python
class AdvancedDetector(MLDetectorPlugin):
    def __init__(self):
        super().__init__()
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check for required dependencies."""
        try:
            import tensorflow as tf
            self.tf_available = True
        except ImportError:
            self.tf_available = False
            print("Warning: TensorFlow not available. Using fallback algorithm.")
```

### Plugin Metadata

Provide rich metadata for your plugins:

```python
class WellDocumentedPlugin(MLDetectorPlugin):
    @property
    def metadata(self) -> dict:
        return {
            'author': 'Your Name',
            'email': 'your.email@example.com',
            'license': 'MIT',
            'tags': ['anomaly-detection', 'machine-learning'],
            'requirements': ['scikit-learn>=1.0.0', 'numpy>=1.20.0'],
            'documentation_url': 'https://github.com/user/plugin-docs'
        }
```

## Testing Guidelines

### Unit Tests

Create comprehensive unit tests for your plugins:

```python
import unittest
import pandas as pd
from unittest.mock import Mock, patch
from plugins.your_plugin import YourDetector

class TestYourDetector(unittest.TestCase):
    def setUp(self):
        self.detector = YourDetector()
        self.sample_data = pd.DataFrame({
            'timestamp': ['2023-01-01 10:00:00', '2023-01-01 10:01:00'],
            'level': ['INFO', 'ERROR'],
            'message': ['Normal message', 'Error occurred']
        })
    
    def test_detection_basic(self):
        """Test basic anomaly detection."""
        result = self.detector.detect_anomalies(self.sample_data)
        self.assertIn('anomaly', result.columns)
        self.assertIn('anomaly_score', result.columns)
    
    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        empty_df = pd.DataFrame()
        result = self.detector.detect_anomalies(empty_df)
        self.assertTrue(result.empty)
    
    def test_invalid_data(self):
        """Test handling of invalid data."""
        invalid_df = pd.DataFrame({'invalid': [1, 2, 3]})
        with self.assertRaises(ValueError):
            self.detector.detect_anomalies(invalid_df)
    
    @patch('your_plugin.some_external_dependency')
    def test_external_dependency(self, mock_dependency):
        """Test with mocked external dependency."""
        mock_dependency.return_value = Mock()
        result = self.detector.detect_anomalies(self.sample_data)
        self.assertIsNotNone(result)
```

### Integration Tests

Test plugin integration with the main system:

```python
class TestPluginIntegration(unittest.TestCase):
    def test_plugin_registration(self):
        """Test plugin registration with the system."""
        from logguard_ml.plugins import PluginManager
        
        manager = PluginManager()
        manager.register_plugin('your_detector', YourDetector)
        
        self.assertIn('your_detector', manager.list_plugins())
    
    def test_plugin_execution(self):
        """Test plugin execution through the system."""
        from logguard_ml.core.ml_model import MLModel
        
        model = MLModel()
        model.register_detector(YourDetector())
        
        result = model.detect_anomalies(self.sample_data)
        self.assertIsNotNone(result)
```

## Performance Considerations

### Memory Efficiency

```python
def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
    """Memory-efficient anomaly detection."""
    # Process data in chunks for large datasets
    chunk_size = 10000
    results = []
    
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        chunk_result = self._process_chunk(chunk)
        results.append(chunk_result)
    
    return pd.concat(results, ignore_index=True)
```

### Caching

```python
from functools import lru_cache

class CachingDetector(MLDetectorPlugin):
    @lru_cache(maxsize=128)
    def _expensive_computation(self, message_hash):
        """Cache expensive computations."""
        # Expensive operation here
        pass
```

### Vectorization

```python
def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
    """Vectorized feature extraction."""
    # Use vectorized operations instead of loops
    features = np.column_stack([
        df['message'].str.len().values,  # Vectorized string operations
        df['level'].map(self.level_encoding).values,  # Vectorized mapping
        pd.to_datetime(df['timestamp']).dt.hour.values  # Vectorized datetime
    ])
    return features
```

## Best Practices

1. **Error Handling**: Always handle edge cases and provide meaningful error messages
2. **Documentation**: Document your plugin thoroughly with docstrings and examples
3. **Configuration**: Make your plugin configurable through parameters
4. **Testing**: Write comprehensive unit and integration tests
5. **Performance**: Consider memory usage and processing time for large datasets
6. **Dependencies**: Minimize external dependencies and handle missing dependencies gracefully
7. **Validation**: Validate input data and configuration parameters
8. **Logging**: Use appropriate logging for debugging and monitoring
9. **Versioning**: Follow semantic versioning for your plugins
10. **Compatibility**: Ensure compatibility with different Python and pandas versions

## Troubleshooting

### Common Issues

**Plugin Not Loading**
- Check plugin file is in the correct directory
- Verify class name matches configuration
- Check for syntax errors in plugin file

**Import Errors**
- Ensure all dependencies are installed
- Check Python path configuration
- Verify plugin module structure

**Performance Issues**
- Profile your plugin code
- Use vectorized operations where possible
- Consider chunking large datasets

**Memory Issues**
- Process data in smaller chunks
- Clean up intermediate variables
- Use appropriate data types

### Debugging

Enable debug logging to troubleshoot plugin issues:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class DebuggablePlugin(MLDetectorPlugin):
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.debug(f"Processing {len(df)} records")
        # Your implementation
        logger.debug("Detection completed")
```

## Plugin Distribution

### Packaging

Create a proper Python package for your plugin:

```
your_plugin/
├── setup.py
├── README.md
├── requirements.txt
├── your_plugin/
│   ├── __init__.py
│   └── detector.py
└── tests/
    └── test_detector.py
```

### Publishing

Consider publishing your plugin to PyPI for easy installation:

```python
# setup.py
from setuptools import setup, find_packages

setup(
    name="logguard-ml-your-plugin",
    version="1.0.0",
    description="Your LogGuard ML plugin",
    packages=find_packages(),
    install_requires=[
        "logguard-ml>=1.0.0",
        "pandas>=1.3.0",
        "numpy>=1.20.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
```

## Contributing Plugins

If you've developed a useful plugin, consider contributing it to the LogGuard ML project:

1. Fork the repository
2. Add your plugin to the `plugins/` directory
3. Include comprehensive tests
4. Update documentation
5. Submit a pull request

## Support

For plugin development support:

- Check the API documentation
- Review existing plugin examples
- Ask questions in the project discussions
- Report bugs in the issue tracker

## Plugin Registry

The LogGuard ML community maintains a registry of available plugins:

- **Official Plugins**: Maintained by the core team
- **Community Plugins**: Contributed by the community
- **Third-party Plugins**: Available through PyPI

Visit the plugin registry to discover existing plugins and share your own.
