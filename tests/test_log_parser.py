"""
Unit tests for LogGuard ML
"""

import os
import pandas as pd
import pytest
from logguard_ml.core.log_parser import LogParser
from logguard_ml.core.ml_model import AnomalyDetector
import yaml

# Path constants for tests
CONFIG_PATH = os.path.join("logguard_ml", "config", "config.yaml")
SAMPLE_LOG_PATH = os.path.join("data", "sample_log.log")

@pytest.fixture
def config():
    """
    Load the YAML config as a Python dictionary.
    """
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def test_parse_sample_log(config):
    """
    Test that sample log file is parsed correctly into a DataFrame.
    """
    parser = LogParser(config)
    df = parser.parse_log_file(SAMPLE_LOG_PATH)

    # Ensure the DataFrame is not empty
    assert not df.empty, "Parsed DataFrame should not be empty"

    # Check required columns exist
    for col in ["timestamp", "level", "message"]:
        assert col in df.columns

    # Check there is at least one ERROR level entry
    assert "ERROR" in df["level"].values

def test_empty_file(config, tmp_path):
    """
    Test that an empty log file returns an empty DataFrame with expected columns.
    """
    empty_log = tmp_path / "empty.log"
    empty_log.write_text("")

    parser = LogParser(config)
    df = parser.parse_log_file(str(empty_log))

    # Should have correct columns but zero rows
    assert list(df.columns) == ["timestamp", "level", "message"]
    assert df.shape[0] == 0

def test_anomaly_detector(config):
    """
    Test anomaly detection logic on a small synthetic DataFrame.
    """
    # Create a small fake DataFrame
    data = {
        "timestamp": ["2025-07-13 10:00:00", "2025-07-13 10:05:00"],
        "level": ["INFO", "ERROR"],
        "message": ["Test log 1", "Critical error happened"]
    }
    df = pd.DataFrame(data)

    detector = AnomalyDetector(config)
    df_out = detector.detect_anomalies(df)

    # Should contain new columns
    assert "message_length" in df_out.columns
    assert "severity_score" in df_out.columns
    assert "is_anomaly" in df_out.columns

    # Values should be either 0 or 1 in is_anomaly
    assert set(df_out["is_anomaly"].unique()).issubset({0, 1})