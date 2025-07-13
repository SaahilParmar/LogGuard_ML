"""
Command Line Interface for LogGuard ML

This module provides a comprehensive CLI for log analysis and anomaly detection.
Supports various output formats, configuration options, and advanced features.

Example:
    $ logguard analyze logs/app.log --ml --output reports/
    $ logguard analyze logs/ --config custom_config.yaml --format json
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import yaml

from logguard_ml import __version__
from logguard_ml.core.log_parser import LogParser, LogParsingError
from logguard_ml.core.ml_model import AnomalyDetector, AnomalyDetectionError
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
        version=f"LogGuard ML {__version__}"
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
                detector = AnomalyDetector(config=config)
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


def main() -> int:
    """
    Main entry point for the CLI.
    
    Returns:
        Exit code
    """
    parser = create_argument_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    if args.command == "analyze":
        return analyze_command(args)
    
    # Should not reach here
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
