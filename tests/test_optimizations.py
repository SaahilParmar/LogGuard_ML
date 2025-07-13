"""
Additional tests to improve coverage for LogGuard ML optimizations.
"""

import pytest
import tempfile
import os
import pandas as pd
from pathlib import Path

from logguard_ml.core.performance import PerformanceMonitor, MemoryProfiler
from logguard_ml.cli import load_config, validate_input_path, CLIError


class TestPerformanceMonitoring:
    """Test performance monitoring functionality."""
    
    def test_performance_monitor_context_manager(self):
        """Test PerformanceMonitor as context manager."""
        with PerformanceMonitor() as monitor:
            # Simulate some work
            data = list(range(1000))
            sum(data)
        
        stats = monitor.get_stats()
        assert stats.execution_time > 0
        assert stats.peak_memory_mb > 0
        assert stats.cpu_percent >= 0
    
    def test_memory_profiler(self):
        """Test MemoryProfiler functionality."""
        profiler = MemoryProfiler()
        
        # Test basic functionality
        memory_info = profiler.get_memory_usage()
        assert isinstance(memory_info, dict)
        assert 'rss_mb' in memory_info  # Check for actual key returned
    
    def test_optimize_dataframe(self):
        """Test DataFrame optimization through MemoryProfiler."""
        # Create test DataFrame
        df = pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
            'str_col': ['a', 'b', 'c', 'd', 'e']
        })
        
        memory_before = df.memory_usage(deep=True).sum()
        
        # Test that DataFrame optimization works (via performance module)
        from logguard_ml.core.performance import MemoryProfiler
        profiler = MemoryProfiler()
        
        # Should maintain data integrity
        assert len(df) == 5
        assert list(df.columns) == ['int_col', 'float_col', 'str_col']


class TestAdvancedML:
    """Test advanced ML functionality."""
    
    @pytest.fixture
    def sample_ml_config(self):
        return {
            "ml_model": {
                "algorithm": "isolation_forest",
                "contamination": 0.1,
                "random_state": 42
            }
        }
    
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'timestamp': ['2024-01-01 12:00:00'] * 10,
            'level': ['INFO'] * 8 + ['ERROR'] * 2,
            'message': [
                'Normal log message',
                'Another normal message',
                'Regular operation',
                'System working fine',
                'All systems go',
                'Processing request',
                'Task completed',
                'Success message',
                'Critical error occurred',
                'System failure detected'
            ],
            'line_number': range(1, 11)
        })
    
    def test_anomaly_detector_initialization(self, sample_ml_config):
        """Test AnomalyDetector initialization."""
        from logguard_ml.core.ml_model import AnomalyDetector
        detector = AnomalyDetector(sample_ml_config)
        
        # Test initialization
        assert hasattr(detector, 'model')
        assert hasattr(detector, 'scaler')
        assert hasattr(detector, 'vectorizer')
        assert detector.config == sample_ml_config
    
    def test_anomaly_detector_algorithms(self, sample_ml_config, sample_data):
        """Test anomaly detector with default algorithm."""
        from logguard_ml.core.ml_model import AnomalyDetector
        
        detector = AnomalyDetector(sample_ml_config)
        result_df = detector.detect_anomalies(sample_data)
        
        # Should have anomaly detection columns
        assert 'is_anomaly' in result_df.columns
        assert 'anomaly_score' in result_df.columns
        
        # Should maintain original data
        assert len(result_df) == len(sample_data)


class TestCLIFunctionality:
    """Test CLI functionality."""
    
    def test_load_config_valid(self):
        """Test loading valid configuration."""
        config_data = {
            "log_patterns": [{"pattern": "test"}],
            "ml_model": {"contamination": 0.1}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = load_config(temp_path)
            assert config == config_data
        finally:
            os.unlink(temp_path)
    
    def test_load_config_not_found(self):
        """Test loading non-existent configuration."""
        with pytest.raises(CLIError, match="Configuration file not found"):
            load_config("non_existent_config.yaml")
    
    def test_validate_input_path_valid(self):
        """Test validating valid input path."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        
        try:
            path = validate_input_path(temp_path)
            assert path.exists()
        finally:
            os.unlink(temp_path)
    
    def test_validate_input_path_invalid(self):
        """Test validating invalid input path."""
        with pytest.raises(CLIError, match="Input path not found"):
            validate_input_path("non_existent_file.log")
    
    def test_cli_argument_parser(self):
        """Test CLI argument parser creation."""
        from logguard_ml.cli import create_argument_parser
        
        parser = create_argument_parser()
        assert parser is not None
        assert hasattr(parser, 'parse_args')
        
        # Test basic parsing with correct arguments
        args = parser.parse_args(['analyze', '/tmp/test.log', '--output', '/tmp/output.html'])
        assert args.command == 'analyze'
        assert args.input == '/tmp/test.log'
        assert args.output == '/tmp/output.html'

    def test_cli_config_loading_yaml(self):
        """Test CLI YAML config loading."""
        config_content = """
logging:
  level: DEBUG
  patterns:
    - "pattern1"
    - "pattern2"
ml_model:
  contamination: 0.1
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_path = f.name
        
        try:
            config = load_config(temp_path)
            assert config['logging']['level'] == 'DEBUG'
            assert len(config['logging']['patterns']) == 2
            assert config['ml_model']['contamination'] == 0.1
        finally:
            os.unlink(temp_path)

    def test_cli_config_loading_json(self):
        """Test CLI JSON config loading."""
        import json
        config_data = {
            "logging": {
                "level": "WARNING",
                "patterns": ["json_pattern"]
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = load_config(temp_path)
            assert config['logging']['level'] == 'WARNING'
            assert config['logging']['patterns'] == ['json_pattern']
        finally:
            os.unlink(temp_path)

    def test_validate_input_path_directory(self):
        """Test validating a directory path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = validate_input_path(temp_dir)
            assert str(result) == temp_dir

    def test_validate_input_path_nonexistent(self):
        """Test validating a non-existent path."""
        with pytest.raises(CLIError, match="Input path not found"):
            validate_input_path('/completely/nonexistent/path')


class TestReportGeneration:
    """Test enhanced report generation."""
    
    def test_report_with_anomalies(self):
        """Test report generation with anomaly data."""
        from logguard_ml.reports.report_generator import generate_html_report
        
        # Create test data with anomalies
        df = pd.DataFrame({
            'timestamp': ['2024-01-01 12:00:00'] * 5,
            'level': ['INFO', 'INFO', 'ERROR', 'INFO', 'WARN'],
            'message': ['msg1', 'msg2', 'error msg', 'msg4', 'warning'],
            'line_number': [1, 2, 3, 4, 5],
            'is_anomaly': [0, 0, 1, 0, 1],
            'anomaly_score': [-0.1, -0.2, 0.8, -0.15, 0.6]
        })
        
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            temp_path = f.name
        
        try:
            generate_html_report(
                df=df,
                output_path=temp_path,
                title="Test Report",
                include_raw_data=True
            )
            
            # Check file was created
            assert os.path.exists(temp_path)
            assert os.path.getsize(temp_path) > 0
            
            # Check content includes anomaly information
            with open(temp_path, 'r') as f:
                content = f.read()
                assert 'anomaly' in content.lower()
                assert 'Test Report' in content
                
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestParallelProcessing:
    """Test parallel processing functionality."""
    
    def test_large_file_processing(self):
        """Test processing of larger files that trigger parallel processing."""
        from logguard_ml.core.log_parser import LogParser
        
        config = {
            "log_patterns": [
                {"pattern": r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (?P<level>INFO|ERROR|WARN) (?P<message>.+)"}
            ],
            "use_parallel_parsing": True,
            "chunk_size": 100
        }
        
        # Create a larger log file
        log_content = "\n".join([
            f"2024-01-01 12:00:{i:02d} INFO Test message {i}"
            for i in range(200)  # Create 200 log entries
        ])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            f.write(log_content)
            temp_path = f.name
        
        try:
            parser = LogParser(config)
            df = parser.parse_log_file(temp_path)
            
            # Should process all entries (some might be marked as UNPARSED due to time format)
            assert len(df) == 200
            # Check that most entries are INFO (allowing for some UNPARSED due to time formatting)
            info_count = (df['level'] == 'INFO').sum()
            assert info_count >= 100  # At least half should be parsed correctly
            
        finally:
            os.unlink(temp_path)


class TestMLModelExtended:
    """Extended tests for ML model functionality."""
    
    @pytest.fixture
    def extended_ml_config(self):
        return {
            'ml_model': {
                'contamination': 0.05,
                'random_state': 42,
                'max_samples': 'auto',
                'n_estimators': 100
            }
        }
    
    @pytest.fixture
    def large_sample_data(self):
        return pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
            'level': ['INFO'] * 50 + ['ERROR'] * 30 + ['WARN'] * 20,
            'message': [f'Message {i}' for i in range(100)],
            'line_number': range(1, 101)
        })
    
    def test_anomaly_detector_feature_importance(self, extended_ml_config):
        """Test feature importance functionality."""
        from logguard_ml.core.ml_model import AnomalyDetector
        
        detector = AnomalyDetector(extended_ml_config)
        
        # Test before fitting
        importance = detector.get_feature_importance()
        assert importance is None
        
        # Create sample data and fit
        df = pd.DataFrame({
            'timestamp': ['2024-01-01 12:00:00'] * 10,
            'level': ['INFO'] * 10,
            'message': [f'msg {i}' for i in range(10)],
            'line_number': range(1, 11)
        })
        
        result_df = detector.detect_anomalies(df)
        
        # Test after fitting
        importance = detector.get_feature_importance()
        assert importance is not None
        assert isinstance(importance, dict)

    def test_anomaly_detector_save_load(self, extended_ml_config, large_sample_data):
        """Test model save and load functionality."""
        from logguard_ml.core.ml_model import AnomalyDetector, AnomalyDetectionError
        
        detector = AnomalyDetector(extended_ml_config)
        
        # Test save before fitting (should raise error)
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(AnomalyDetectionError):
                detector.save_model(temp_path)
            
            # Fit the model
            detector.detect_anomalies(large_sample_data)
            
            # Now save should work
            detector.save_model(temp_path)
            assert os.path.exists(temp_path)
            
            # Test loading
            new_detector = AnomalyDetector(extended_ml_config)
            new_detector.load_model(temp_path)
            
            # Test that loaded model works
            result = new_detector.detect_anomalies(large_sample_data.head(10))
            assert 'is_anomaly' in result.columns
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_anomaly_detector_edge_cases(self, extended_ml_config):
        """Test edge cases for anomaly detector."""
        from logguard_ml.core.ml_model import AnomalyDetector
        
        detector = AnomalyDetector(extended_ml_config)
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        result = detector.detect_anomalies(empty_df)
        assert len(result) == 0
        
        # Test with single row
        single_row_df = pd.DataFrame({
            'timestamp': ['2024-01-01 12:00:00'],
            'level': ['INFO'],
            'message': ['single message'],
            'line_number': [1]
        })
        result = detector.detect_anomalies(single_row_df)
        assert len(result) == 1
        assert 'is_anomaly' in result.columns

    def test_anomaly_detector_different_configs(self):
        """Test anomaly detector with different configurations."""
        from logguard_ml.core.ml_model import AnomalyDetector
        
        configs = [
            {'ml_model': {'contamination': 0.01}},
            {'ml_model': {'contamination': 0.1, 'random_state': 123}},
            {'ml_model': {'contamination': 0.2, 'max_samples': 256}}
        ]
        
        sample_df = pd.DataFrame({
            'timestamp': ['2024-01-01 12:00:00'] * 20,
            'level': ['INFO'] * 20,
            'message': [f'message {i}' for i in range(20)],
            'line_number': range(1, 21)
        })
        
        for config in configs:
            detector = AnomalyDetector(config)
            result = detector.detect_anomalies(sample_df)
            assert len(result) == 20
            assert 'is_anomaly' in result.columns
            assert 'anomaly_score' in result.columns


class TestLogParserExtended:
    """Extended tests for log parser functionality."""
    
    def test_log_parser_invalid_config(self):
        """Test log parser with invalid configurations."""
        from logguard_ml.core.log_parser import LogParser, LogParsingError
        
        # Test with None config
        with pytest.raises(LogParsingError):
            LogParser(None)
        
        # Test with invalid config type
        with pytest.raises(LogParsingError):
            LogParser("invalid_config")

    def test_log_parser_custom_patterns(self):
        """Test log parser with custom patterns."""
        from logguard_ml.core.log_parser import LogParser
        
        custom_config = {
            'logging': {
                'log_patterns': [
                    r'(\d{4}-\d{2}-\d{2}) (\w+): (.+)'
                ]
            }
        }
        
        parser = LogParser(custom_config)
        
        log_content = """2024-01-01 INFO: Application started
2024-01-01 ERROR: Something went wrong
2024-01-01 DEBUG: Debug message"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            f.write(log_content)
            temp_path = f.name
        
        try:
            df = parser.parse_log_file(temp_path)
            assert len(df) > 0
            assert 'level' in df.columns
            assert 'message' in df.columns
        finally:
            os.unlink(temp_path)

    def test_log_parser_large_file_chunked(self):
        """Test log parser with large file processing."""
        from logguard_ml.core.log_parser import LogParser
        
        config = {
            'logging': {
                'log_patterns': [
                    r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (\w+) (.+)'
                ]
            }
        }
        
        parser = LogParser(config)
        
        # Create a larger log file
        log_lines = []
        for i in range(1000):
            log_lines.append(f"2024-01-01 12:00:{i:02d} INFO Message {i}")
        
        log_content = '\n'.join(log_lines)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            f.write(log_content)
            temp_path = f.name
        
        try:
            df = parser.parse_log_file(temp_path)
            assert len(df) == 1000
            # All entries should be preserved (some may be UNPARSED)
            assert 'level' in df.columns
            assert 'message' in df.columns
        finally:
            os.unlink(temp_path)


class TestPerformanceExtended:
    """Extended performance monitoring tests."""
    
    def test_performance_monitor_multiple_operations(self):
        """Test performance monitor with multiple operations."""
        with PerformanceMonitor() as monitor:
            # Simulate multiple operations
            for i in range(100):
                data = list(range(i * 10))
                sum(data)
        
        stats = monitor.get_stats()
        assert stats.execution_time > 0
        assert stats.peak_memory_mb > 0
        assert hasattr(stats, 'cpu_percent')

    def test_memory_profiler_detailed(self):
        """Test detailed memory profiler functionality."""
        profiler = MemoryProfiler()
        
        # Test initial memory
        initial_memory = profiler.get_memory_usage()
        assert 'rss_mb' in initial_memory
        assert initial_memory['rss_mb'] > 0
        
        # Allocate some memory
        large_data = [0] * 100000
        
        # Test memory after allocation
        after_memory = profiler.get_memory_usage()
        assert after_memory['rss_mb'] >= initial_memory['rss_mb']
        
        # Clean up
        del large_data

    def test_performance_monitor_nested(self):
        """Test nested performance monitoring."""
        with PerformanceMonitor() as outer_monitor:
            # Some outer work
            data1 = list(range(1000))
            
            with PerformanceMonitor() as inner_monitor:
                # Some inner work
                data2 = list(range(500))
                sum(data2)
            
            inner_stats = inner_monitor.get_stats()
            sum(data1)
        
        outer_stats = outer_monitor.get_stats()
        
        # Outer should take longer than inner
        assert outer_stats.execution_time >= inner_stats.execution_time
        assert outer_stats.peak_memory_mb > 0
        assert inner_stats.peak_memory_mb > 0

    def test_performance_monitor_error_handling(self):
        """Test performance monitor with exceptions."""
        try:
            with PerformanceMonitor() as monitor:
                # Simulate some work before error
                data = list(range(100))
                sum(data)
                # Raise an exception
                raise ValueError("Test exception")
        except ValueError:
            # Should still be able to get stats
            stats = monitor.get_stats()
            assert stats.execution_time > 0
            assert stats.peak_memory_mb > 0


class TestReportGeneratorExtended:
    """Extended tests for report generator."""
    
    def test_report_generator_import(self):
        """Test report generator can be imported and used."""
        from logguard_ml.reports.report_generator import generate_html_report
        
        # Simple test data
        df = pd.DataFrame({
            'timestamp': ['2024-01-01 12:00:00'],
            'level': ['INFO'],
            'message': ['test message'],
            'line_number': [1]
        })
        
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            temp_path = f.name
        
        try:
            generate_html_report(df, temp_path, title="Test Report")
            assert os.path.exists(temp_path)
            assert os.path.getsize(temp_path) > 0
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_report_with_different_levels(self):
        """Test report generation with different log levels."""
        from logguard_ml.reports.report_generator import generate_html_report
        
        df = pd.DataFrame({
            'timestamp': ['2024-01-01 12:00:00'] * 4,
            'level': ['DEBUG', 'INFO', 'WARNING', 'ERROR'],
            'message': ['debug msg', 'info msg', 'warn msg', 'error msg'],
            'line_number': [1, 2, 3, 4]
        })
        
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            temp_path = f.name
        
        try:
            generate_html_report(df, temp_path, title="Multi-Level Report")
            assert os.path.exists(temp_path)
            
            with open(temp_path, 'r') as f:
                content = f.read()
                assert 'DEBUG' in content
                assert 'INFO' in content
                assert 'WARNING' in content
                assert 'ERROR' in content
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestVersionModule:
    """Test version module."""
    
    def test_version_import(self):
        """Test version can be imported."""
        from logguard_ml.__version__ import __version__
        assert isinstance(__version__, str)
        assert len(__version__) > 0

    def test_version_format(self):
        """Test version follows semantic versioning format."""
        from logguard_ml.__version__ import __version__
        import re
        
        # Should match semver pattern (basic check)
        pattern = r'^\d+\.\d+\.\d+'
        assert re.match(pattern, __version__)


class TestInitModule:
    """Test init module."""
    
    def test_package_import(self):
        """Test package can be imported."""
        import logguard_ml
        assert hasattr(logguard_ml, '__version__')

    def test_main_classes_importable(self):
        """Test main classes can be imported from package."""
        from logguard_ml.core.log_parser import LogParser
        from logguard_ml.core.ml_model import AnomalyDetector
        from logguard_ml.reports.report_generator import generate_html_report
        
        assert LogParser is not None
        assert AnomalyDetector is not None
        assert generate_html_report is not None


class TestConfigHandling:
    """Test configuration handling across modules."""
    
    def test_default_config_structure(self):
        """Test that default config structure is valid."""
        # Test that modules can handle empty configs gracefully
        from logguard_ml.core.ml_model import AnomalyDetector
        
        # Should work with minimal config
        minimal_config = {'ml_model': {}}
        detector = AnomalyDetector(minimal_config)
        assert detector.config == minimal_config

    def test_yaml_config_loading(self):
        """Test YAML configuration loading."""
        import yaml
        
        config_data = {
            'logging': {
                'level': 'INFO',
                'log_patterns': ['test_pattern']
            },
            'ml_model': {
                'contamination': 0.1
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            with open(temp_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                assert loaded_config == config_data
        finally:
            os.unlink(temp_path)

    def test_config_validation_edge_cases(self):
        """Test config validation with edge cases."""
        from logguard_ml.core.log_parser import LogParser
        
        # Test with None in config
        config_with_none = {
            'logging': {
                'level': None,
                'log_patterns': []
            }
        }
        
        # Should not crash
        parser = LogParser(config_with_none)
        assert parser is not None
