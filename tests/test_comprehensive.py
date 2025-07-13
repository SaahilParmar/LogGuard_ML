"""
Comprehensive test suite for LogGuard ML

Tests cover log parsing, anomaly detection, and report generation
with various edge cases and error conditions.
"""

import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path

from logguard_ml.core.log_parser import LogParser, LogParsingError
from logguard_ml.core.ml_model import AnomalyDetector, AnomalyDetectionError
from logguard_ml.reports.report_generator import generate_html_report, ReportGenerationError


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "log_patterns": [
            {
                "pattern": r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (?P<level>ERROR|WARN|INFO) (?P<message>.+)"
            }
        ],
        "ml_model": {
            "contamination": 0.1,
            "random_state": 42,
            "max_samples": "auto"
        }
    }


@pytest.fixture
def sample_log_content():
    """Sample log content for testing."""
    return """2024-01-01 12:00:00 INFO Application started
2024-01-01 12:01:00 INFO User login: john_doe
2024-01-01 12:02:00 WARN Database connection slow
2024-01-01 12:03:00 ERROR Failed to process order #12345
2024-01-01 12:04:00 ERROR Critical system failure
2024-01-01 12:05:00 INFO Application shutdown"""


@pytest.fixture
def temp_log_file(sample_log_content):
    """Create a temporary log file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
        f.write(sample_log_content)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    os.unlink(temp_path)


class TestLogParser:
    """Test cases for LogParser class."""
    
    def test_parse_sample_log(self, sample_config, temp_log_file):
        """Test parsing a sample log file."""
        parser = LogParser(config=sample_config)
        df = parser.parse_log_file(temp_log_file)
        
        assert len(df) == 6
        assert list(df.columns) == ["timestamp", "level", "message", "line_number"]
        assert df["level"].tolist() == ["INFO", "INFO", "WARN", "ERROR", "ERROR", "INFO"]
        
    def test_empty_file(self, sample_config):
        """Test parsing an empty log file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            temp_path = f.name
        
        try:
            parser = LogParser(config=sample_config)
            df = parser.parse_log_file(temp_path)
            
            assert len(df) == 0
            assert list(df.columns) == ["timestamp", "level", "message"]
        finally:
            os.unlink(temp_path)
    
    def test_file_not_found(self, sample_config):
        """Test handling of non-existent file."""
        parser = LogParser(config=sample_config)
        
        with pytest.raises(FileNotFoundError):
            parser.parse_log_file("non_existent_file.log")
    
    def test_invalid_config(self):
        """Test handling of invalid configuration."""
        with pytest.raises(LogParsingError):
            LogParser(config="invalid_config")
    
    def test_no_patterns_config(self):
        """Test configuration without log patterns."""
        config = {"other_settings": {}}
        parser = LogParser(config=config)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            f.write("some log line")
            temp_path = f.name
        
        try:
            df = parser.parse_log_file(temp_path)
            # Should have 1 unparsed entry
            assert len(df) == 1
            assert df["level"].iloc[0] == "UNPARSED"
        finally:
            os.unlink(temp_path)
    
    def test_get_supported_fields(self, sample_config):
        """Test getting supported fields from patterns."""
        parser = LogParser(config=sample_config)
        fields = parser.get_supported_fields()
        
        assert "timestamp" in fields
        assert "level" in fields
        assert "message" in fields


class TestAnomalyDetector:
    """Test cases for AnomalyDetector class."""
    
    def test_anomaly_detection(self, sample_config, temp_log_file):
        """Test basic anomaly detection functionality."""
        # First parse the logs
        parser = LogParser(config=sample_config)
        df = parser.parse_log_file(temp_log_file)
        
        # Then detect anomalies
        detector = AnomalyDetector(config=sample_config)
        result_df = detector.detect_anomalies(df)
        
        assert "is_anomaly" in result_df.columns
        assert "anomaly_score" in result_df.columns
        assert "message_length" in result_df.columns
        assert "severity_score" in result_df.columns
        
        # Check that anomaly flags are binary
        assert result_df["is_anomaly"].isin([0, 1]).all()
    
    def test_empty_dataframe(self, sample_config):
        """Test anomaly detection with empty DataFrame."""
        detector = AnomalyDetector(config=sample_config)
        empty_df = pd.DataFrame()
        
        result_df = detector.detect_anomalies(empty_df)
        
        assert "is_anomaly" in result_df.columns
        assert "anomaly_score" in result_df.columns
        assert len(result_df) == 0
    
    def test_invalid_contamination(self):
        """Test invalid contamination parameter."""
        config = {
            "ml_model": {
                "contamination": 1.5  # Invalid value > 0.5
            }
        }
        
        with pytest.raises(AnomalyDetectionError):
            AnomalyDetector(config=config)
    
    def test_minimal_dataframe(self, sample_config):
        """Test with minimal DataFrame."""
        detector = AnomalyDetector(config=sample_config)
        
        df = pd.DataFrame({
            "message": ["Short", "This is a much longer message that should be different"],
            "level": ["INFO", "ERROR"]
        })
        
        result_df = detector.detect_anomalies(df)
        
        assert len(result_df) == 2
        assert "is_anomaly" in result_df.columns


class TestReportGeneration:
    """Test cases for report generation."""
    
    def test_generate_html_report(self, sample_config, temp_log_file):
        """Test HTML report generation."""
        # Parse logs and detect anomalies
        parser = LogParser(config=sample_config)
        df = parser.parse_log_file(temp_log_file)
        
        detector = AnomalyDetector(config=sample_config)
        df = detector.detect_anomalies(df)
        
        # Generate report
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            output_path = f.name
        
        try:
            generate_html_report(df, output_path)
            
            # Check that file was created and has content
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
            
            # Check for HTML content
            with open(output_path, 'r') as f:
                content = f.read()
                assert "<!DOCTYPE html>" in content
                assert "LogGuard ML" in content
                
        finally:
            os.unlink(output_path)
    
    def test_report_with_empty_data(self):
        """Test report generation with empty data."""
        empty_df = pd.DataFrame()
        
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            output_path = f.name
        
        try:
            generate_html_report(empty_df, output_path)
            
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
            
        finally:
            os.unlink(output_path)


class TestIntegration:
    """Integration tests for the complete workflow."""
    
    def test_complete_workflow(self, sample_config, temp_log_file):
        """Test the complete log analysis workflow."""
        # Step 1: Parse logs
        parser = LogParser(config=sample_config)
        df = parser.parse_log_file(temp_log_file)
        
        assert len(df) > 0
        
        # Step 2: Detect anomalies
        detector = AnomalyDetector(config=sample_config)
        df = detector.detect_anomalies(df)
        
        assert "is_anomaly" in df.columns
        
        # Step 3: Generate report
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            output_path = f.name
        
        try:
            generate_html_report(df, output_path)
            
            assert os.path.exists(output_path)
            
        finally:
            os.unlink(output_path)
    
    def test_workflow_with_malformed_logs(self, sample_config):
        """Test workflow with malformed log entries."""
        malformed_content = """This is not a valid log line
2024-01-01 12:00:00 INFO This is valid
Another invalid line
2024-01-01 12:01:00 ERROR This is also valid"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            f.write(malformed_content)
            temp_path = f.name
        
        try:
            parser = LogParser(config=sample_config)
            df = parser.parse_log_file(temp_path)
            
            # Should have 4 total lines (2 valid + 2 unparsed)
            assert len(df) == 4
            
            # Check valid lines
            valid_df = df[df["level"] != "UNPARSED"]
            assert len(valid_df) == 2
            assert valid_df["level"].tolist() == ["INFO", "ERROR"]
            
            # Check unparsed lines
            unparsed_df = df[df["level"] == "UNPARSED"]
            assert len(unparsed_df) == 2
            
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__])
