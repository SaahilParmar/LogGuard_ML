"""
Tests for monitoring module to improve coverage.

This test module focuses on testing the monitoring functionality including
real-time log monitoring, alert management, and streaming processors.
"""

import pytest
import tempfile
import os
import time
import threading
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import pandas as pd
from queue import Queue

from logguard_ml.core.monitoring import (
    LogFileHandler, AlertManager, StreamProcessor, LogMonitor
)


class TestLogFileHandler:
    """Test LogFileHandler functionality."""
    
    def test_init(self):
        """Test LogFileHandler initialization."""
        callback = MagicMock()
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"initial content")
            temp_path = Path(f.name)
        
        try:
            handler = LogFileHandler(callback, temp_path)
            assert handler.callback == callback
            assert handler.target_file == temp_path
            assert handler.last_position > 0
        finally:
            os.unlink(temp_path)
    
    def test_init_nonexistent_file(self):
        """Test LogFileHandler with non-existent file."""
        callback = MagicMock()
        temp_path = Path("/tmp/nonexistent.log")
        
        handler = LogFileHandler(callback, temp_path)
        assert handler.last_position == 0
    
    def test_on_modified(self):
        """Test file modification handling."""
        callback = MagicMock()
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            handler = LogFileHandler(callback, temp_path)
            
            # Mock event
            event = MagicMock()
            event.is_directory = False
            event.src_path = str(temp_path)
            
            handler.on_modified(event)
            callback.assert_called_once()
        finally:
            os.unlink(temp_path)
    
    def test_on_modified_directory(self):
        """Test directory modification (should be ignored)."""
        callback = MagicMock()
        temp_path = Path("/tmp/test.log")
        
        handler = LogFileHandler(callback, temp_path)
        
        # Mock directory event
        event = MagicMock()
        event.is_directory = True
        event.src_path = "/tmp/"
        
        handler.on_modified(event)
        callback.assert_not_called()
    
    def test_on_modified_different_file(self):
        """Test modification of different file (should be ignored)."""
        callback = MagicMock()
        temp_path = Path("/tmp/target.log")
        
        handler = LogFileHandler(callback, temp_path)
        
        # Mock different file event
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/tmp/other.log"
        
        handler.on_modified(event)
        callback.assert_not_called()


class TestAlertManager:
    """Test AlertManager functionality."""
    
    def test_init_disabled(self):
        """Test AlertManager initialization with alerts disabled."""
        config = {"alerting": {"enabled": False}}
        manager = AlertManager(config)
        
        assert manager.enabled is False
        assert manager.anomaly_threshold == 5  # default
        assert manager.time_window == 5  # default
    
    def test_init_enabled_with_config(self):
        """Test AlertManager initialization with custom config."""
        config = {
            "alerting": {
                "enabled": True,
                "anomaly_threshold": 10,
                "time_window_minutes": 15,
                "throttle_minutes": 30,
                "email": {
                    "smtp_server": "smtp.example.com",
                    "smtp_port": 587,
                    "username": "user@example.com",
                    "password": "password",
                    "recipients": ["admin@example.com"]
                },
                "webhook": {
                    "url": "http://example.com/webhook",
                    "method": "POST"
                }
            }
        }
        
        manager = AlertManager(config)
        assert manager.enabled is True
        assert manager.anomaly_threshold == 10
        assert manager.time_window == 15
        assert manager.alert_throttle == 30
        assert manager.email_config["smtp_server"] == "smtp.example.com"
        assert manager.webhook_config["url"] == "http://example.com/webhook"
    
    def test_should_alert_disabled(self):
        """Test should_alert when alerts are disabled."""
        config = {"alerting": {"enabled": False}}
        manager = AlertManager(config)
        
        assert manager.should_alert(10) is False
    
    def test_should_alert_below_threshold(self):
        """Test should_alert when below threshold."""
        config = {
            "alerting": {
                "enabled": True,
                "anomaly_threshold": 10
            }
        }
        manager = AlertManager(config)
        
        assert manager.should_alert(5) is False
    
    def test_should_alert_above_threshold(self):
        """Test should_alert when above threshold."""
        config = {
            "alerting": {
                "enabled": True,
                "anomaly_threshold": 5
            }
        }
        manager = AlertManager(config)
        
        assert manager.should_alert(10) is True
    
    @patch('logguard_ml.core.monitoring.smtplib.SMTP')
    def test_send_email_alert(self, mock_smtp):
        """Test sending email alert."""
        config = {
            "alerting": {
                "enabled": True,
                "email": {
                    "enabled": True,
                    "smtp_server": "smtp.example.com",
                    "smtp_port": 587,
                    "username": "user@example.com",
                    "password": "password",
                    "from": "alerts@example.com",
                    "to": ["admin@example.com"]
                }
            }
        }
        
        manager = AlertManager(config)
        mock_server = MagicMock()
        mock_smtp.return_value = mock_server
        
        anomaly_data = {"count": 5, "timestamp": "2024-01-01 12:00:00"}
        manager._send_email_alert(anomaly_data)
        
        mock_smtp.assert_called_once_with("smtp.example.com", 587)
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with("user@example.com", "password")
        mock_server.send_message.assert_called_once()
        mock_server.quit.assert_called_once()
    
    @patch('logguard_ml.core.monitoring.requests.post')
    def test_send_webhook_alert(self, mock_post):
        """Test sending webhook alert."""
        config = {
            "alerting": {
                "enabled": True,
                "webhook": {
                    "enabled": True,
                    "url": "http://example.com/webhook",
                    "headers": {"Content-Type": "application/json"}
                }
            }
        }
        
        manager = AlertManager(config)
        mock_post.return_value.status_code = 200
        
        anomaly_data = {"count": 3, "timestamp": "2024-01-01 12:00:00"}
        manager._send_webhook_alert(anomaly_data)
        
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "http://example.com/webhook"
        assert "json" in kwargs
        assert kwargs["json"]["anomaly_count"] == 3
    
    def test_send_alert_throttling(self):
        """Test alert throttling functionality."""
        config = {
            "alerting": {
                "enabled": True,
                "anomaly_threshold": 5,
                "throttle_minutes": 1
            }
        }
        manager = AlertManager(config)
        
        # First alert should be sent
        assert manager.should_alert(10) is True
        
        # Add to recent alerts to simulate throttling
        from datetime import datetime
        manager.recent_alerts.append(datetime.now())
        
        # Second alert should be throttled
        assert manager.should_alert(10) is False
    
    def test_format_alert_message(self):
        """Test alert message formatting."""
        config = {"alerting": {"enabled": True}}
        manager = AlertManager(config)
        
        anomaly_data = {
            "count": 5,
            "timestamp": "2024-01-01 12:00:00",
            "sample_messages": ["Error message 1", "Error message 2"]
        }
        
        message = manager._format_alert_message(anomaly_data)
        assert "LogGuard ML Anomaly Alert" in message
        assert "5" in message  # anomaly count
        assert "Error message 1" in message


class TestStreamProcessor:
    """Test StreamProcessor functionality."""
    
    def test_init(self):
        """Test StreamProcessor initialization."""
        config = {"test": "config"}
        processor = StreamProcessor(config, buffer_size=100)
        
        assert processor.config == config
        assert processor.buffer_size == 100
        assert processor.log_queue.maxsize == 100
        assert processor.is_running is False
    
    def test_add_log_entry(self):
        """Test adding log entry to queue."""
        config = {}
        processor = StreamProcessor(config)
        
        processor.add_log_entry("test log line")
        assert processor.log_queue.qsize() == 1
    
    def test_add_log_entry_full_queue(self):
        """Test adding log entry to full queue."""
        config = {}
        processor = StreamProcessor(config, buffer_size=1)
        
        # Fill the queue
        processor.add_log_entry("first entry")
        
        # This should not block or raise an exception
        processor.add_log_entry("second entry")
        assert processor.log_queue.qsize() == 1
    
    @patch('logguard_ml.core.monitoring.LogParser')
    @patch('logguard_ml.core.monitoring.AdvancedAnomalyDetector')
    def test_process_batch(self, mock_detector, mock_parser):
        """Test processing a batch of log entries."""
        # Mock parser and detector instances
        mock_parser_instance = MagicMock()
        mock_parser_instance.parse_log_lines.return_value = pd.DataFrame({
            'timestamp': ['2024-01-01 12:00:00'],
            'level': ['INFO'],
            'message': ['test message']
        })
        mock_parser.return_value = mock_parser_instance
        
        mock_detector_instance = MagicMock()
        mock_detector_instance.detect_anomalies.return_value = (
            pd.DataFrame({'is_anomaly': [0]}), 
            pd.DataFrame({'anomaly_score': [0.1]})
        )
        mock_detector.return_value = mock_detector_instance
        
        # Create processor after mocks are set up
        config = {}
        processor = StreamProcessor(config)
        
        log_lines = ["2024-01-01 12:00:00 INFO test message"]
        result = processor.process_batch(log_lines)
        
        assert result is not None
        mock_parser_instance.parse_log_lines.assert_called_once_with(log_lines)
        mock_detector_instance.detect_anomalies.assert_called_once()
    
    def test_start_stop(self):
        """Test starting and stopping processor."""
        config = {}
        processor = StreamProcessor(config)
        
        # Start processor
        processor.start()
        assert processor.is_running is True
        
        # Stop processor
        processor.stop()
        assert processor.is_running is False


class TestLogMonitor:
    """Test LogMonitor functionality."""
    
    def test_init(self):
        """Test LogMonitor initialization."""
        config = {"test": "config"}
        log_path = "/tmp/test.log"
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        
        try:
            monitor = LogMonitor(config, temp_path)
            assert monitor.config == config
            assert str(monitor.log_path) == temp_path
            assert monitor.is_monitoring is False
        finally:
            os.unlink(temp_path)
    
    def test_init_nonexistent_file(self):
        """Test LogMonitor with non-existent file."""
        config = {}
        
        with pytest.raises(FileNotFoundError):
            LogMonitor(config, "/tmp/nonexistent.log")
    
    @patch('logguard_ml.core.monitoring.Observer')
    def test_start_monitoring(self, mock_observer):
        """Test starting monitoring."""
        config = {}
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        
        try:
            monitor = LogMonitor(config, temp_path)
            mock_observer_instance = MagicMock()
            mock_observer.return_value = mock_observer_instance
            
            monitor.start_monitoring()
            
            assert monitor.is_monitoring is True
            mock_observer_instance.start.assert_called_once()
        finally:
            os.unlink(temp_path)
    
    @patch('logguard_ml.core.monitoring.Observer')
    def test_stop_monitoring(self, mock_observer):
        """Test stopping monitoring."""
        config = {}
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        
        try:
            monitor = LogMonitor(config, temp_path)
            mock_observer_instance = MagicMock()
            mock_observer.return_value = mock_observer_instance
            
            # Start then stop
            monitor.start_monitoring()
            monitor.stop_monitoring()
            
            assert monitor.is_monitoring is False
            mock_observer_instance.stop.assert_called_once()
            mock_observer_instance.join.assert_called_once()
        finally:
            os.unlink(temp_path)
    
    def test_read_new_lines(self):
        """Test reading new lines from file."""
        config = {}
        
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
            f.write("line 1\nline 2\n")
            f.flush()
            temp_path = f.name
        
        try:
            monitor = LogMonitor(config, temp_path)
            
            # Append new lines
            with open(temp_path, 'a') as f:
                f.write("line 3\nline 4\n")
            
            new_lines = monitor.read_new_lines()
            assert len(new_lines) == 2
            assert "line 3" in new_lines[0]
            assert "line 4" in new_lines[1]
        finally:
            os.unlink(temp_path)
    
    def test_read_new_lines_empty(self):
        """Test reading new lines when file hasn't changed."""
        config = {}
        
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
            f.write("line 1\nline 2\n")
            f.flush()
            temp_path = f.name
        
        try:
            monitor = LogMonitor(config, temp_path)
            
            # Read once to set position
            monitor.read_new_lines()
            
            # Read again without changes
            new_lines = monitor.read_new_lines()
            assert len(new_lines) == 0
        finally:
            os.unlink(temp_path)
    
    @patch('logguard_ml.core.monitoring.AlertManager')
    def test_process_anomalies(self, mock_alert_manager):
        """Test processing anomalies."""
        # Mock alert manager instance first
        mock_manager = MagicMock()
        mock_manager.should_alert.return_value = True
        mock_alert_manager.return_value = mock_manager
        
        config = {"alerting": {"enabled": True}}
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        
        try:
            monitor = LogMonitor(config, temp_path)
            
            # Create test data with anomalies
            df = pd.DataFrame({
                'timestamp': ['2024-01-01 12:00:00'] * 3,
                'level': ['INFO', 'ERROR', 'WARN'],
                'message': ['msg1', 'error', 'warning'],
                'is_anomaly': [0, 1, 1]
            })
            
            monitor.process_anomalies(df)
            
            # Should have called alert manager
            mock_manager.should_alert.assert_called_once_with(2)  # 2 anomalies
        finally:
            os.unlink(temp_path)
