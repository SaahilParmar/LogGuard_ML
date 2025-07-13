#!/usr/bin/env python3
"""
LogGuard ML - Legacy Main Entry Point

This script maintains backward compatibility while directing users
to the new CLI interface. It provides the same functionality as before
but with improved error handling and logging.

Usage:
    python main.py --logfile data/sample_log.log --ml
    
Recommended:
    Use the new CLI: logguard analyze data/sample_log.log --ml
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

import yaml

from logguard_ml.core.log_parser import LogParser, LogParsingError
from logguard_ml.core.ml_model import AnomalyDetector, AnomalyDetectionError
from logguard_ml.reports.report_generator import generate_html_report, ReportGenerationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """
    Load YAML configuration into a Python dictionary.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        SystemExit: If configuration cannot be loaded
    """
    try:
        config_path = Path(config_path)
        if not config_path.exists():
            # Try legacy path
            legacy_path = Path("config/config.yaml")
            if legacy_path.exists():
                config_path = legacy_path
                logger.warning(f"Using legacy config path: {config_path}")
            else:
                print(f"ERROR: Configuration file not found: {config_path}")
                sys.exit(1)
                
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            
        logger.info(f"Configuration loaded from: {config_path}")
        return config
        
    except yaml.YAMLError as e:
        print(f"ERROR: Invalid YAML configuration: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load configuration: {e}")
        sys.exit(1)


def main():
    """Main entry point with enhanced error handling and user guidance."""
    
    # Show deprecation warning
    warnings.warn(
        "main.py is deprecated. Use the new CLI: 'logguard analyze <logfile> --ml'",
        DeprecationWarning,
        stacklevel=2
    )
    
    parser = argparse.ArgumentParser(
        description="LogGuard ML - Log Analysis and Anomaly Detection Tool (Legacy Interface)",
        epilog="Recommended: Use 'logguard analyze <logfile> --ml' for the improved CLI"
    )
    parser.add_argument(
        "--logfile",
        type=str,
        required=True,
        help="Path to log file to analyze",
    )
    parser.add_argument(
        "--ml",
        action="store_true",
        help="Enable ML-based anomaly detection",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="logguard_ml/config/config.yaml",
        help="Path to YAML config file (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/anomaly_report.html",
        help="Path to output HTML report (default: %(default)s)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Check if logfile exists
        logfile_path = Path(args.logfile)
        if not logfile_path.exists():
            print(f"ERROR: Log file not found: {args.logfile}")
            print("💡 Tip: Make sure the file path is correct and accessible")
            sys.exit(1)

        # Load YAML configuration
        print("📁 Loading configuration...")
        config = load_config(args.config)

        # Initialize log parser
        print("🔍 Initializing log parser...")
        parser_obj = LogParser(config=config)
        df = parser_obj.parse_log_file(args.logfile)

        if df.empty:
            print("⚠️  No log entries were parsed. Check your log patterns and file format.")
            print("💡 Tip: Verify that your log format matches the patterns in config.yaml")
            sys.exit(1)

        print(f"✅ Parsed {len(df)} log entries from {args.logfile}")

        # Initialize ML anomaly detector (optional)
        if args.ml:
            print("🤖 Running ML anomaly detection...")
            try:
                detector = AnomalyDetector(config=config)
                df = detector.detect_anomalies(df)
                anomaly_count = df["is_anomaly"].sum()
                percentage = (anomaly_count / len(df)) * 100
                print(f"🔍 Anomalies detected: {anomaly_count} ({percentage:.2f}%)")
            except AnomalyDetectionError as e:
                print(f"❌ Anomaly detection failed: {e}")
                sys.exit(1)
        else:
            df["is_anomaly"] = 0
            print("⏭️  ML anomaly detection skipped")

        # Generate HTML report
        print("📊 Generating HTML report...")
        try:
            # Ensure output directory exists
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            generate_html_report(df, output_path=str(output_path))
            print(f"✅ Report saved to: {output_path}")
            
            # Show additional guidance
            print("\\n🎉 Analysis complete!")
            print(f"📖 Open {output_path} in your browser to view the results")
            print("\\n💡 For more features, try the new CLI:")
            print(f"   logguard analyze {args.logfile} --ml --verbose")
            
        except ReportGenerationError as e:
            print(f"❌ Report generation failed: {e}")
            sys.exit(1)

    except LogParsingError as e:
        print(f"❌ Log parsing failed: {e}")
        print("💡 Tip: Check your log file format and configuration patterns")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\\n⏹️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"💥 Unexpected error: {e}")
        print("💡 Tip: Run with --verbose for more details")
        sys.exit(1)


if __name__ == "__main__":
    main()
