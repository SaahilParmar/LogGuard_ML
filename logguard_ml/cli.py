"""
Enhanced Command Line Interface for LogGuard ML

This module provides a comprehensive CLI for log analysis, anomaly detection,
real-time monitoring, and performance optimization features.

Features:
- Traditional batch analysis with enhanced performance
- Real-time log monitoring with alerting
- Performance profiling and optimization
- Multiple output formats with caching
- Advanced ML algorithms and ensemble methods

Example:
    $ logguard analyze logs/app.log --ml --algorithm ensemble
    $ logguard monitor logs/app.log --alerts
    $ logguard analyze logs/ --config custom_config.yaml --parallel
"""

import argparse
import logging
import sys
import signal
from pathlib import Path
from typing import Optional

import yaml

from logguard_ml.utils.version import get_version, get_system_version_info
from logguard_ml.core.log_parser import LogParser, LogParsingError
from logguard_ml.core.advanced_ml import AdvancedAnomalyDetector, AnomalyDetectionError
from logguard_ml.core.monitoring import LogMonitor
from logguard_ml.core.performance import optimize_pandas_settings, PerformanceMonitor
from logguard_ml.reports.report_generator import generate_html_report, ReportGenerationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CLIError(Exception):
    """Custom exception for CLI errors."""
    pass


def load_config(config_path: str) -> dict:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        CLIError: If configuration cannot be loaded
    """
    try:
        config_path = Path(config_path)
        if not config_path.exists():
            raise CLIError(f"Configuration file not found: {config_path}")
            
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            
        if not isinstance(config, dict):
            raise CLIError("Configuration must be a valid YAML dictionary")
            
        logger.info(f"Configuration loaded from: {config_path}")
        return config
        
    except yaml.YAMLError as e:
        raise CLIError(f"Invalid YAML configuration: {e}")
    except Exception as e:
        raise CLIError(f"Error loading configuration: {e}")


def validate_input_path(input_path: str) -> Path:
    """
    Validate and return input path.
    
    Args:
        input_path: Input file or directory path
        
    Returns:
        Validated Path object
        
    Raises:
        CLIError: If path is invalid
    """
    path = Path(input_path)
    if not path.exists():
        raise CLIError(f"Input path not found: {path}")
    return path


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        prog="logguard",
        description="LogGuard ML - AI-Powered Log Analysis & Anomaly Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  logguard analyze app.log --ml
  logguard analyze logs/ --config custom.yaml --output reports/
  logguard analyze app.log --ml --format json --verbose
  
For more information, visit: https://github.com/SaahilParmar/LogGuard_ML
        """
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"LogGuard ML {get_version()}"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze log files for anomalies"
    )
    
    analyze_parser.add_argument(
        "input",
        help="Path to log file or directory containing log files"
    )
    
    analyze_parser.add_argument(
        "--ml",
        action="store_true",
        help="Enable ML-based anomaly detection"
    )
    
    analyze_parser.add_argument(
        "--config",
        type=str,
        default="logguard_ml/config/config.yaml",
        help="Path to YAML configuration file (default: %(default)s)"
    )
    
    analyze_parser.add_argument(
        "--output",
        type=str,
        default="reports/anomaly_report.html",
        help="Output path for the generated report (default: %(default)s)"
    )
    
    analyze_parser.add_argument(
        "--format",
        choices=["html", "json", "csv"],
        default="html",
        help="Output format (default: %(default)s)"
    )
    
    analyze_parser.add_argument(
        "--title",
        type=str,
        help="Custom title for the report"
    )
    
    analyze_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    analyze_parser.add_argument(
        "--no-raw-data",
        action="store_true",
        help="Exclude raw data table from HTML reports"
    )
    
    analyze_parser.add_argument(
        "--algorithm",
        choices=["isolation_forest", "one_class_svm", "local_outlier_factor", "ensemble"],
        default="isolation_forest",
        help="ML algorithm to use for anomaly detection"
    )
    
    analyze_parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel processing for large files"
    )
    
    analyze_parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="Chunk size for parallel processing"
    )
    
    analyze_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching for report generation"
    )
    
    analyze_parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable performance profiling"
    )
    
    # Monitor command (new)
    monitor_parser = subparsers.add_parser(
        "monitor",
        help="Real-time log monitoring with anomaly detection"
    )
    
    monitor_parser.add_argument(
        "log_file",
        help="Path to log file to monitor in real-time"
    )
    
    monitor_parser.add_argument(
        "--config",
        type=str,
        default="logguard_ml/config/config.yaml",
        help="Path to YAML configuration file"
    )
    
    monitor_parser.add_argument(
        "--alerts",
        action="store_true",
        help="Enable alerting for detected anomalies"
    )
    
    monitor_parser.add_argument(
        "--buffer-size",
        type=int,
        default=100,
        help="Stream buffer size for real-time processing"
    )
    
    monitor_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    # Profile command (new)
    profile_parser = subparsers.add_parser(
        "profile",
        help="Performance profiling of log analysis operations"
    )
    
    profile_parser.add_argument(
        "input",
        help="Path to log file for profiling"
    )
    
    profile_parser.add_argument(
        "--config",
        type=str,
        default="logguard_ml/config/config.yaml",
        help="Path to YAML configuration file"
    )
    
    profile_parser.add_argument(
        "--operations",
        nargs="+",
        choices=["parse", "ml", "report"],
        default=["parse", "ml", "report"],
        help="Operations to profile"
    )

    return parser


def analyze_command(args) -> int:
    """
    Execute the analyze command.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Set logging level
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Verbose logging enabled")
        
        # Validate inputs
        input_path = validate_input_path(args.input)
        
        # Load configuration
        config = load_config(args.config)
        
        # Initialize log parser
        logger.info("Initializing log parser...")
        parser = LogParser(config=config)
        
        # Parse log file(s)
        logger.info(f"Parsing logs from: {input_path}")
        df = parser.parse_log_file(input_path)
        
        if df.empty:
            logger.warning("No log entries were parsed. Check your log patterns and file format.")
            print("⚠️  No log entries were parsed.")
            return 1
        
        logger.info(f"Successfully parsed {len(df)} log entries")
        print(f"✅ Parsed {len(df)} log entries from {input_path}")
        
        # ML anomaly detection (optional)
        if args.ml:
            logger.info("Initializing anomaly detection...")
            try:
                detector = AdvancedAnomalyDetector(config=config)
                df = detector.detect_anomalies(df)
                
                anomaly_count = df["is_anomaly"].sum()
                anomaly_percentage = (anomaly_count / len(df)) * 100
                
                logger.info(f"Detected {anomaly_count} anomalies ({anomaly_percentage:.2f}%)")
                print(f"🔍 Anomaly detection complete: {anomaly_count} anomalies detected ({anomaly_percentage:.2f}%)")
                
            except AnomalyDetectionError as e:
                logger.error(f"Anomaly detection failed: {e}")
                print(f"❌ Anomaly detection failed: {e}")
                return 1
        else:
            df["is_anomaly"] = 0
            print("⏭️  ML anomaly detection skipped")
        
        # Generate report
        logger.info(f"Generating {args.format} report...")
        
        if args.format == "html":
            title = args.title or "LogGuard ML - Anomaly Detection Report"
            generate_html_report(
                df=df,
                output_path=args.output,
                title=title,
                include_raw_data=not args.no_raw_data
            )
        elif args.format == "json":
            output_path = Path(args.output).with_suffix(".json")
            df.to_json(output_path, orient="records", indent=2)
        elif args.format == "csv":
            output_path = Path(args.output).with_suffix(".csv")
            df.to_csv(output_path, index=False)
        
        print(f"📊 Report saved to: {args.output}")
        logger.info(f"Report generated successfully: {args.output}")
        
        return 0
        
    except (CLIError, LogParsingError, ReportGenerationError) as e:
        logger.error(f"Command failed: {e}")
        print(f"❌ {e}")
        return 1
    except KeyboardInterrupt:
        print("\\n⏹️  Operation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"💥 Unexpected error: {e}")
        return 1


def monitor_command(args) -> int:
    """
    Execute the real-time monitoring command.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Set logging level
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Verbose logging enabled")
        
        # Load configuration
        config = load_config(args.config)
        
        # Update config with command line options
        if hasattr(args, 'buffer_size'):
            config.setdefault('stream_buffer_size', args.buffer_size)
        
        if args.alerts:
            config.setdefault('alerting', {})['enabled'] = True
        
        # Initialize and start monitor
        print(f"🔍 Starting real-time monitoring of {args.log_file}")
        monitor = LogMonitor(config, args.log_file)
        
        # Set up signal handler for graceful shutdown
        def signal_handler(signum, frame):
            print("\n🛑 Stopping monitoring...")
            monitor.stop_monitoring()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start monitoring
        monitor.start_monitoring()
        
        print("✅ Monitoring started. Press Ctrl+C to stop.")
        print(f"📊 Status: {monitor.get_status()}")
        
        # Keep running until interrupted
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping monitoring...")
            monitor.stop_monitoring()
        
        return 0
        
    except Exception as e:
        logger.error(f"Monitoring failed: {e}")
        print(f"❌ Monitoring failed: {e}")
        return 1


def profile_command(args) -> int:
    """
    Execute the performance profiling command.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Load configuration
        config = load_config(args.config)
        
        print(f"🔬 Profiling operations on {args.input}")
        
        with PerformanceMonitor() as global_monitor:
            # Profile parsing
            if "parse" in args.operations:
                print("📝 Profiling log parsing...")
                with PerformanceMonitor() as parse_monitor:
                    parser = LogParser(config)
                    df = parser.parse_log_file(args.input)
                
                parse_stats = parse_monitor.get_stats()
                print(f"  - Parsed {len(df)} entries in {parse_stats.execution_time:.2f}s")
                print(f"  - Peak memory: {parse_stats.peak_memory_mb:.1f}MB")
                print(f"  - Throughput: {len(df)/parse_stats.execution_time:.0f} entries/sec")
            
            # Profile ML
            if "ml" in args.operations and 'df' in locals():
                print("🧠 Profiling ML anomaly detection...")
                with PerformanceMonitor() as ml_monitor:
                    detector = AdvancedAnomalyDetector(config)
                    df_with_anomalies = detector.detect_anomalies(df)
                
                ml_stats = ml_monitor.get_stats()
                anomaly_count = df_with_anomalies['is_anomaly'].sum()
                print(f"  - Detected {anomaly_count} anomalies in {ml_stats.execution_time:.2f}s")
                print(f"  - Peak memory: {ml_stats.peak_memory_mb:.1f}MB")
                print(f"  - ML throughput: {len(df)/ml_stats.execution_time:.0f} entries/sec")
            
            # Profile report generation
            if "report" in args.operations and 'df_with_anomalies' in locals():
                print("📊 Profiling report generation...")
                with PerformanceMonitor() as report_monitor:
                    temp_output = "temp_profile_report.html"
                    generate_html_report(df_with_anomalies, temp_output)
                    
                    # Clean up temp file
                    import os
                    if os.path.exists(temp_output):
                        os.remove(temp_output)
                
                report_stats = report_monitor.get_stats()
                print(f"  - Generated report in {report_stats.execution_time:.2f}s")
                print(f"  - Peak memory: {report_stats.peak_memory_mb:.1f}MB")
        
        # Overall stats
        global_stats = global_monitor.get_stats()
        print(f"\n📈 Overall Performance:")
        print(f"  - Total time: {global_stats.execution_time:.2f}s")
        print(f"  - Peak memory: {global_stats.peak_memory_mb:.1f}MB")
        print(f"  - CPU usage: {global_stats.cpu_percent:.1f}%")
        print(f"  - Memory usage: {global_stats.memory_percent:.1f}%")
        
        return 0
        
    except Exception as e:
        logger.error(f"Profiling failed: {e}")
        print(f"❌ Profiling failed: {e}")
        return 1


def main() -> int:
    """Main entry point for the CLI."""
    # Optimize pandas settings for performance
    optimize_pandas_settings()
    
    parser = create_argument_parser()
    try:
        args = parser.parse_args()
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
        return 130
    
    if args.command is None:
        parser.print_help()
        return 1
    
    try:
        if args.command == "analyze":
            return analyze_command(args)
        elif args.command == "monitor":
            return monitor_command(args)
        elif args.command == "profile":
            return profile_command(args)
        else:
            print(f"❌ Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"💥 Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
