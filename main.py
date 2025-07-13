#!/usr/bin/env python3
"""
LogGuard ML - Main entry point

This script ties together:
- log parsing
- optional ML anomaly detection
- report generation

Run from the command line, e.g.:
    python main.py --logfile data/sample_log.log --ml
"""

import argparse
import yaml
import os
import sys
import pandas as pd

from utils.log_parser import LogParser
from utils.ml_model import AnomalyDetector
from reports.report_generator import generate_html_report

def load_config(config_path: str) -> dict:
    """
    Loads YAML config into a Python dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(
        description="LogGuard ML - Log Analysis and Anomaly Detection Tool"
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
        default="config/config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/anomaly_report.html",
        help="Path to output HTML report",
    )

    args = parser.parse_args()

    # Check if logfile exists
    if not os.path.exists(args.logfile):
        print(f"ERROR: Log file {args.logfile} not found.")
        sys.exit(1)

    # Load YAML configuration
    config = load_config(args.config)

    # Initialize log parser
    parser_obj = LogParser(config=config)
    df = parser_obj.parse_log_file(args.logfile)

    print(f"[+] Parsed {len(df)} log entries from {args.logfile}")

    # Initialize ML anomaly detector (optional)
    if args.ml:
        detector = AnomalyDetector(config=config)
        df = detector.detect_anomalies(df)
        anomaly_count = df["is_anomaly"].sum()
        print(f"[+] Anomalies detected: {anomaly_count}")
    else:
        df["is_anomaly"] = 0
        print("[+] ML anomaly detection skipped.")

    # Generate HTML report
    generate_html_report(df, output_path=args.output)
    print(f"[+] Report saved to: {args.output}")

if __name__ == "__main__":
    main()
