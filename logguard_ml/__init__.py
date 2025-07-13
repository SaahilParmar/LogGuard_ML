"""
LogGuard ML - AI-Powered Log Analysis & Anomaly Detection Framework

A comprehensive toolkit for intelligent log analysis and anomaly detection using
machine learning algorithms.

Key Features:
- Intelligent log parsing with configurable regex patterns
- ML-based anomaly detection using Isolation Forest
- Beautiful HTML report generation with visualizations
- Configurable and extensible architecture
- Command-line interface for easy usage

Example:
    >>> from logguard_ml import LogParser, AnomalyDetector
    >>> parser = LogParser(config)
    >>> df = parser.parse_log_file("app.log")
    >>> detector = AnomalyDetector(config)
    >>> anomalies = detector.detect_anomalies(df)
"""

from logguard_ml.__version__ import __version__
from logguard_ml.core.log_parser import LogParser
from logguard_ml.core.ml_model import AnomalyDetector
from logguard_ml.reports.report_generator import generate_html_report

__all__ = [
    "__version__",
    "LogParser", 
    "AnomalyDetector",
    "generate_html_report",
]
