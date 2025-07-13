"""
Log Parser for LogGuard ML

This class reads log files line by line,
applies regex patterns from config.yaml,
and returns a DataFrame of structured log entries.
"""

import re
import pandas as pd
from typing import List, Dict

class LogParser:
    def __init__(self, config: Dict):
        """
        Initialize parser with config dictionary.

        Args:
            config (dict): Configuration dictionary from config.yaml
        """
        self.patterns = []
        if "log_patterns" in config:
            for pat in config["log_patterns"]:
                self.patterns.append(re.compile(pat["pattern"]))

    def parse_log_file(self, filepath: str) -> pd.DataFrame:
        """
        Parse log file into DataFrame.

        Args:
            filepath (str): Path to log file

        Returns:
            pd.DataFrame: DataFrame of parsed logs
        """
        records = []

        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                for regex in self.patterns:
                    match = regex.match(line)
                    if match:
                        record = match.groupdict()
                        records.append(record)
                        break  # stop at first matching regex

        if records:
            df = pd.DataFrame(records)
        else:
            # Return empty DataFrame with expected columns
            df = pd.DataFrame(columns=["timestamp", "level", "message"])

        return df