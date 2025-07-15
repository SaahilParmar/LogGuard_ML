"""
LogGuard ML - AI-Powered Log Analysis & Anomaly Detection Framework

A comprehensive toolkit for intelligent log analysis and anomaly detection using
machine learning algorithms with enhanced performance optimization.

Key Features:
- Intelligent log parsing with configurable regex patterns and parallel processing
- Advanced ML-based anomaly detection with multiple algorithms and ensemble methods
- Beautiful HTML report generation with interactive visualizations and caching
- Real-time log monitoring with streaming anomaly detection and alerting
- Performance optimization with memory profiling and batch processing
- Configurable and extensible architecture with comprehensive error handling

Example:
    >>> from logguard_ml import LogParser, AdvancedAnomalyDetector, LogMonitor
    >>> parser = LogParser(config)
    >>> df = parser.parse_log_file("app.log")
    >>> detector = AdvancedAnomalyDetector(config)
    >>> anomalies = detector.detect_anomalies(df)
"""

from logguard_ml.__version__ import __version__
from logguard_ml.core.advanced_ml import AdvancedAnomalyDetector, AnomalyDetector
from logguard_ml.core.log_parser import LogParser
from logguard_ml.reports.report_generator import generate_html_report

# Optional imports - only import if dependencies are available
try:
    from logguard_ml.core.monitoring import LogMonitor

    _MONITORING_AVAILABLE = True
except ImportError:
    LogMonitor = None
    _MONITORING_AVAILABLE = False

__all__ = [
    "__version__",
    "LogParser",
    "AdvancedAnomalyDetector",
    "AnomalyDetector",  # Backward compatibility
    "generate_html_report",
]

if _MONITORING_AVAILABLE:
    __all__.append("LogMonitor")
