"""
Log Parser for LogGuard ML

This module provides intelligent log parsing capabilities with configurable
regex patterns and comprehensive error handling.

Classes:
    LogParser: Main class for parsing log files into structured DataFrames

Example:
    >>> from logguard_ml.core.log_parser import LogParser
    >>> config = {"log_patterns": [{"pattern": r"(?P<timestamp>\\d{4}-\\d{2}-\\d{2}).*"}]}
    >>> parser = LogParser(config)
    >>> df = parser.parse_log_file("app.log")
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


class LogParsingError(Exception):
    """Custom exception for log parsing errors."""
    pass


class LogParser:
    """
    Intelligent log parser that extracts structured data from log files.
    
    The LogParser uses configurable regex patterns to extract fields from
    log entries and returns them as a pandas DataFrame for further analysis.
    
    Attributes:
        patterns (List[re.Pattern]): Compiled regex patterns for log parsing
        config (Dict): Configuration dictionary containing parsing rules
    
    Example:
        >>> config = {
        ...     "log_patterns": [
        ...         {"pattern": r"(?P<timestamp>\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}) "
        ...                    r"(?P<level>ERROR|WARN|INFO) (?P<message>.+)"}
        ...     ]
        ... }
        >>> parser = LogParser(config)
        >>> df = parser.parse_log_file("application.log")
    """

    def __init__(self, config: Dict) -> None:
        """
        Initialize parser with configuration dictionary.

        Args:
            config: Configuration dictionary containing log patterns and settings
            
        Raises:
            LogParsingError: If configuration is invalid or patterns cannot be compiled
        """
        self.config = config
        self.patterns: List[re.Pattern] = []
        
        if not isinstance(config, dict):
            raise LogParsingError("Configuration must be a dictionary")
            
        self._compile_patterns()
        logger.info(f"LogParser initialized with {len(self.patterns)} patterns")

    def _compile_patterns(self) -> None:
        """
        Compile regex patterns from configuration.
        
        Raises:
            LogParsingError: If patterns cannot be compiled
        """
        if "log_patterns" not in self.config:
            logger.warning("No log_patterns found in configuration")
            return
            
        for i, pattern_config in enumerate(self.config["log_patterns"]):
            if not isinstance(pattern_config, dict) or "pattern" not in pattern_config:
                logger.warning(f"Skipping invalid pattern configuration at index {i}")
                continue
                
            try:
                compiled_pattern = re.compile(pattern_config["pattern"])
                self.patterns.append(compiled_pattern)
                logger.debug(f"Compiled pattern: {pattern_config['pattern']}")
            except re.error as e:
                raise LogParsingError(f"Invalid regex pattern at index {i}: {e}")

    def parse_log_file(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Parse log file into a structured DataFrame.

        Args:
            filepath: Path to the log file to parse
            
        Returns:
            DataFrame containing parsed log entries with extracted fields
            
        Raises:
            LogParsingError: If file cannot be read or parsing fails
            FileNotFoundError: If the specified file does not exist
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Log file not found: {filepath}")
            
        if not filepath.is_file():
            raise LogParsingError(f"Path is not a file: {filepath}")
            
        logger.info(f"Parsing log file: {filepath}")
        records = []
        
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                        
                    parsed_record = self._parse_line(line, line_num)
                    if parsed_record:
                        records.append(parsed_record)
                        
        except IOError as e:
            raise LogParsingError(f"Error reading file {filepath}: {e}")
            
        if records:
            df = pd.DataFrame(records)
            logger.info(f"Successfully parsed {len(df)} log entries")
        else:
            # Return empty DataFrame with expected columns
            df = pd.DataFrame(columns=["timestamp", "level", "message"])
            logger.warning("No log entries matched any patterns")
            
        return df

    def _parse_line(self, line: str, line_num: int) -> Optional[Dict]:
        """
        Parse a single log line using configured patterns.
        
        Args:
            line: Log line to parse
            line_num: Line number for error reporting
            
        Returns:
            Dictionary of extracted fields or None if no pattern matches
        """
        for pattern in self.patterns:
            match = pattern.match(line)
            if match:
                record = match.groupdict()
                record["line_number"] = line_num
                return record
                
        logger.debug(f"No pattern matched line {line_num}: {line[:50]}...")
        return None

    def get_supported_fields(self) -> List[str]:
        """
        Get list of fields that can be extracted by current patterns.
        
        Returns:
            List of field names that patterns can extract
        """
        fields = set()
        for pattern in self.patterns:
            fields.update(pattern.groupindex.keys())
        return sorted(list(fields))
