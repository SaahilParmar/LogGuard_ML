"""
Extended tests for CLI module to improve coverage.

This test module focuses on testing the CLI functions that are not covered
by existing tests, particularly the main execution functions and error handling.
"""

import pytest
import tempfile
import os
import sys
import subprocess
import time
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import yaml
import json

from logguard_ml.cli import (
    CLIError, load_config, validate_input_path, create_argument_parser,
    analyze_command, monitor_command, profile_command, main
)


class TestCLIExtended:
    """Extended CLI functionality tests."""
    
    def test_load_config_invalid_yaml(self):
        """Test loading invalid YAML configuration."""
        invalid_yaml = "key: value\n  invalid: indentation"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml)
            temp_path = f.name
        
        try:
            with pytest.raises(CLIError, match="Invalid YAML configuration"):
                load_config(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_load_config_non_dict(self):
        """Test loading YAML that's not a dictionary."""
        yaml_list = "- item1\n- item2"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_list)
            temp_path = f.name
        
        try:
            with pytest.raises(CLIError, match="Configuration must be a valid YAML dictionary"):
                load_config(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_load_config_permission_error(self):
        """Test loading config with permission error."""
        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            with pytest.raises(CLIError, match="Error loading configuration"):
                load_config("test.yaml")
    
    def test_create_argument_parser_subcommands(self):
        """Test argument parser with different subcommands."""
        parser = create_argument_parser()
        
        # Test analyze command with all options
        args = parser.parse_args([
            'analyze', '/tmp/test.log',
            '--output', '/tmp/output.html',
            '--config', 'config.yaml',
            '--ml',
            '--algorithm', 'isolation_forest',
            '--parallel',
            '--verbose',
            '--no-cache',
            '--profile'
        ])
        
        assert args.command == 'analyze'
        assert args.input == '/tmp/test.log'
        assert args.output == '/tmp/output.html'
        assert args.config == 'config.yaml'
        assert args.ml is True
        assert args.algorithm == 'isolation_forest'
        assert args.parallel is True
        assert args.verbose is True
        assert args.no_cache is True
        assert args.profile is True
    
    def test_create_argument_parser_monitor_command(self):
        """Test monitor command parsing."""
        parser = create_argument_parser()
        
        args = parser.parse_args([
            'monitor', '/tmp/test.log',
            '--config', 'config.yaml',
            '--alerts',
            '--buffer-size', '200',
            '--verbose'
        ])
        
        assert args.command == 'monitor'
        assert args.log_file == '/tmp/test.log'
        assert args.config == 'config.yaml'
        assert args.alerts is True
        assert args.buffer_size == 200
        assert args.verbose is True
    
    def test_create_argument_parser_profile_command(self):
        """Test profile command parsing."""
        parser = create_argument_parser()
        
        args = parser.parse_args([
            'profile', '/tmp/test.log',
            '--config', 'config.yaml',
            '--operations', 'parse', 'ml'
        ])
        
        assert args.command == 'profile'
        assert args.input == '/tmp/test.log'
        assert args.config == 'config.yaml'
        assert args.operations == ['parse', 'ml']
    
    @patch('logguard_ml.cli.validate_input_path')
    @patch('logguard_ml.cli.load_config')
    @patch('logguard_ml.cli.optimize_pandas_settings')
    @patch('logguard_ml.cli.LogParser')
    @patch('logguard_ml.cli.AdvancedAnomalyDetector')
    @patch('logguard_ml.cli.generate_html_report')
    def test_analyze_command_success(self, mock_report, mock_detector, mock_parser, 
                                   mock_optimize, mock_config, mock_validate):
        """Test successful analyze command execution."""
        # Setup mocks
        mock_validate.return_value = Path('/tmp/test.log')
        mock_config.return_value = {'test': 'config'}
        
        # Mock parser to return valid DataFrame
        import pandas as pd
        mock_df = pd.DataFrame({
            'timestamp': ['2024-01-01 12:00:00'],
            'level': ['INFO'],
            'message': ['test message'],
            'line_number': [1]
        })
        mock_parser_instance = MagicMock()
        mock_parser_instance.parse_log_file.return_value = mock_df
        mock_parser.return_value = mock_parser_instance
        
        # Mock detector to return single DataFrame with anomaly columns
        mock_df_with_anomalies = mock_df.copy()
        mock_df_with_anomalies['is_anomaly'] = [0]
        mock_df_with_anomalies['anomaly_score'] = [0.1]
        mock_detector_instance = MagicMock()
        mock_detector_instance.detect_anomalies.return_value = mock_df_with_anomalies
        mock_detector.return_value = mock_detector_instance
        
        # Create mock arguments
        args = MagicMock()
        args.verbose = False
        args.input = '/tmp/test.log'
        args.config = 'config.yaml'
        args.ml = True
        args.output = '/tmp/output.html'
        args.algorithm = 'isolation_forest'
        args.parallel = False
        args.profile = False
        args.no_cache = False
        
        result = analyze_command(args)
        assert result == 0
        
        mock_validate.assert_called_once_with('/tmp/test.log')
        mock_config.assert_called_once_with('config.yaml')
        mock_parser.assert_called_once()
        mock_detector.assert_called_once()
    
    @patch('logguard_ml.cli.validate_input_path')
    def test_analyze_command_invalid_path(self, mock_validate):
        """Test analyze command with invalid path."""
        mock_validate.side_effect = CLIError("Path not found")
        
        args = MagicMock()
        args.verbose = False
        args.input = '/invalid/path'
        args.config = 'config.yaml'
        
        result = analyze_command(args)
        assert result == 1
    
    @patch('logguard_ml.cli.load_config')
    def test_analyze_command_invalid_config(self, mock_config):
        """Test analyze command with invalid config."""
        mock_config.side_effect = CLIError("Config error")
        
        args = MagicMock()
        args.verbose = False
        args.input = '/tmp/test.log'
        args.config = 'invalid.yaml'
        
        with patch('logguard_ml.cli.validate_input_path', return_value=Path('/tmp/test.log')):
            result = analyze_command(args)
            assert result == 1
    
    @patch('logguard_ml.cli.validate_input_path')
    @patch('logguard_ml.cli.load_config')
    @patch('logguard_ml.cli.LogMonitor')
    @patch('time.sleep')
    def test_monitor_command_success(self, mock_sleep, mock_monitor, mock_config, mock_validate):
        """Test successful monitor command execution."""
        mock_validate.return_value = Path('/tmp/test.log')
        mock_config.return_value = {'test': 'config'}
        mock_monitor_instance = MagicMock()
        mock_monitor_instance.get_status.return_value = "Running"
        mock_monitor.return_value = mock_monitor_instance
        
        # Mock sleep to raise KeyboardInterrupt after first call
        mock_sleep.side_effect = [None, KeyboardInterrupt()]
        
        args = MagicMock()
        args.verbose = False
        args.log_file = '/tmp/test.log'
        args.config = 'config.yaml'
        args.alerts = True
        args.buffer_size = 100
        
        with patch('logguard_ml.cli.signal.signal'):
            result = monitor_command(args)
            assert result == 0
            
        mock_monitor_instance.start_monitoring.assert_called_once()
        mock_monitor_instance.stop_monitoring.assert_called_once()
    
    @patch('logguard_ml.cli.load_config')
    def test_monitor_command_config_error(self, mock_config):
        """Test monitor command with config error."""
        mock_config.side_effect = CLIError("Config error")
        
        args = MagicMock()
        args.verbose = False
        args.log_file = '/tmp/test.log'
        args.config = 'invalid.yaml'
        
        result = monitor_command(args)
        assert result == 1
    
    @patch('logguard_ml.cli.load_config')
    @patch('logguard_ml.cli.LogMonitor')
    def test_monitor_command_monitor_error(self, mock_monitor, mock_config):
        """Test monitor command with LogMonitor error."""
        mock_config.return_value = {'test': 'config'}
        mock_monitor.side_effect = Exception("Monitor initialization failed")
        
        args = MagicMock()
        args.verbose = False
        args.log_file = '/tmp/test.log'
        args.config = 'config.yaml'
        args.alerts = False
        args.buffer_size = 100
        
        result = monitor_command(args)
        assert result == 1
    
    @patch('logguard_ml.cli.validate_input_path')
    @patch('logguard_ml.cli.load_config')
    @patch('logguard_ml.cli.PerformanceMonitor')
    @patch('logguard_ml.cli.LogParser')
    @patch('logguard_ml.cli.AdvancedAnomalyDetector')
    def test_profile_command_success(self, mock_detector, mock_parser, mock_perf, mock_config, mock_validate):
        """Test successful profile command execution."""
        mock_validate.return_value = Path('/tmp/test.log')
        mock_config.return_value = {'test': 'config'}
        
        # Mock PerformanceMonitor context manager with stats
        mock_stats = MagicMock()
        mock_stats.execution_time = 1.5
        mock_stats.peak_memory_mb = 25.5
        mock_stats.cpu_percent = 45.0
        mock_stats.memory_percent = 60.0
        
        mock_perf_instance = MagicMock()
        mock_perf_instance.__enter__.return_value = mock_perf_instance
        mock_perf_instance.__exit__.return_value = None
        mock_perf_instance.get_stats.return_value = mock_stats
        mock_perf.return_value = mock_perf_instance
        
        # Mock parser
        mock_parser_instance = MagicMock()
        import pandas as pd
        mock_df = pd.DataFrame({
            'timestamp': ['2024-01-01 12:00:00'],
            'level': ['INFO'],
            'message': ['test message']
        })
        mock_parser_instance.parse_log_file.return_value = mock_df
        mock_parser.return_value = mock_parser_instance
        
        # Mock detector
        mock_df_with_anomalies = mock_df.copy()
        mock_df_with_anomalies['is_anomaly'] = [0]
        mock_detector_instance = MagicMock()
        mock_detector_instance.detect_anomalies.return_value = mock_df_with_anomalies
        mock_detector.return_value = mock_detector_instance
        
        args = MagicMock()
        args.verbose = False
        args.input = '/tmp/test.log'
        args.config = 'config.yaml'
        args.operations = ['parse', 'ml']
        
        result = profile_command(args)
        assert result == 0
    
    @patch('sys.argv', ['logguard', 'analyze', '/tmp/test.log'])
    @patch('logguard_ml.cli.analyze_command')
    def test_main_analyze_command(self, mock_analyze):
        """Test main function with analyze command."""
        mock_analyze.return_value = 0
        
        with patch('logguard_ml.cli.create_argument_parser') as mock_parser:
            mock_parser_instance = MagicMock()
            mock_args = MagicMock()
            mock_args.command = 'analyze'
            mock_parser_instance.parse_args.return_value = mock_args
            mock_parser.return_value = mock_parser_instance
            
            result = main()
            assert result == 0
            mock_analyze.assert_called_once_with(mock_args)
    
    @patch('sys.argv', ['logguard', 'monitor', '/tmp/test.log'])
    @patch('logguard_ml.cli.monitor_command')
    def test_main_monitor_command(self, mock_monitor):
        """Test main function with monitor command."""
        mock_monitor.return_value = 0
        
        with patch('logguard_ml.cli.create_argument_parser') as mock_parser:
            mock_parser_instance = MagicMock()
            mock_args = MagicMock()
            mock_args.command = 'monitor'
            mock_parser_instance.parse_args.return_value = mock_args
            mock_parser.return_value = mock_parser_instance
            
            result = main()
            assert result == 0
            mock_monitor.assert_called_once_with(mock_args)
    
    @patch('sys.argv', ['logguard', 'profile', '/tmp/test.log'])
    @patch('logguard_ml.cli.profile_command')
    def test_main_profile_command(self, mock_profile):
        """Test main function with profile command."""
        mock_profile.return_value = 0
        
        with patch('logguard_ml.cli.create_argument_parser') as mock_parser:
            mock_parser_instance = MagicMock()
            mock_args = MagicMock()
            mock_args.command = 'profile'
            mock_parser_instance.parse_args.return_value = mock_args
            mock_parser.return_value = mock_parser_instance
            
            result = main()
            assert result == 0
            mock_profile.assert_called_once_with(mock_args)
    
    @patch('sys.argv', ['logguard', 'unknown'])
    def test_main_unknown_command(self):
        """Test main function with unknown command."""
        with patch('logguard_ml.cli.create_argument_parser') as mock_parser:
            mock_parser_instance = MagicMock()
            mock_args = MagicMock()
            mock_args.command = 'unknown'
            mock_parser_instance.parse_args.return_value = mock_args
            mock_parser.return_value = mock_parser_instance
            
            result = main()
            assert result == 1
    
    @patch('sys.argv', ['logguard', '--version'])
    def test_main_version(self):
        """Test main function with version argument."""
        with patch('logguard_ml.cli.create_argument_parser') as mock_parser:
            mock_parser_instance = MagicMock()
            mock_parser_instance.parse_args.side_effect = SystemExit(0)
            mock_parser.return_value = mock_parser_instance
            
            with pytest.raises(SystemExit):
                main()
    
    def test_main_keyboard_interrupt(self):
        """Test main function handling keyboard interrupt."""
        with patch('logguard_ml.cli.create_argument_parser') as mock_parser:
            mock_parser_instance = MagicMock()
            mock_parser_instance.parse_args.side_effect = KeyboardInterrupt()
            mock_parser.return_value = mock_parser_instance
            
            result = main()
            assert result == 130  # Standard exit code for SIGINT
