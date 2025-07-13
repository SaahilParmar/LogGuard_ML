"""
Optimized Log Parser for LogGuard ML

This module provides intelligent log parsing capabilities with configurable
regex patterns, parallel processing, and comprehensive error handling.

Classes:
    LogParser: Main class for parsing log files into structured DataFrames
    ParallelLogParser: High-performance parser with parallel processing

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
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

import pandas as pd

from .performance import (
    PerformanceMonitor, 
    BatchProcessor, 
    MemoryProfiler, 
    profile_function,
    get_optimal_chunk_size
)

logger = logging.getLogger(__name__)


class LogParsingError(Exception):
    """Custom exception for log parsing errors."""
    pass


class LogParser:
    """
    Intelligent log parser that extracts structured data from log files.
    
    The LogParser uses configurable regex patterns to extract fields from
    log entries and returns them as a pandas DataFrame for further analysis.
    Enhanced with performance monitoring and memory optimization.
    
    Attributes:
        patterns (List[re.Pattern]): Compiled regex patterns for log parsing
        config (Dict): Configuration dictionary containing parsing rules
        use_parallel (bool): Whether to use parallel processing for large files
        chunk_size (int): Size of chunks for batch processing
    
    Example:
        >>> config = {
        ...     "log_patterns": [
        ...         {"pattern": r"(?P<timestamp>\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}) "
        ...                    r"(?P<level>ERROR|WARN|INFO) (?P<message>.+)"}
        ...     ]
        ... }
        >>> parser = LogParser(config)
        >>> df = parser.parse_log_file("app.log")
    """

    def __init__(self, config: Dict) -> None:
        """
        Initialize parser with configuration dictionary.

        Args:
            config: Configuration dictionary containing log patterns and settings
            
        Raises:
            LogParsingError: If configuration is invalid or patterns cannot be compiled
        """
        if not isinstance(config, dict):
            raise LogParsingError("Configuration must be a dictionary")
            
        self.config = config
        self.patterns: List[re.Pattern] = []
        self.use_parallel = config.get("use_parallel_parsing", True)
        self.chunk_size = config.get("chunk_size", 10000)
        self.max_workers = config.get("max_workers", min(4, mp.cpu_count()))
            
        self._compile_patterns()
        logger.info(f"LogParser initialized with {len(self.patterns)} patterns")
        if self.use_parallel:
            logger.info(f"Parallel processing enabled: chunk_size={self.chunk_size}, max_workers={self.max_workers}")

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

    @profile_function
    def parse_log_file(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Parse log file into structured DataFrame with performance optimization.

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
        
        with PerformanceMonitor() as monitor:
            try:
                # Get file size for optimization decisions
                file_size_mb = filepath.stat().st_size / (1024 * 1024)
                available_memory = MemoryProfiler.get_memory_usage()['available_mb']
                
                logger.debug(f"File size: {file_size_mb:.1f}MB, Available memory: {available_memory:.1f}MB")
                
                # Choose processing strategy based on file size
                if file_size_mb > 100 or (file_size_mb > 10 and self.use_parallel):
                    # Use batch processing for large files
                    df = self._parse_large_file(filepath, file_size_mb, available_memory)
                else:
                    # Use traditional processing for small files
                    df = self._parse_small_file(filepath)
                
                # Optimize memory usage
                if not df.empty:
                    df = MemoryProfiler.optimize_dataframe(df)
                
                # Log performance stats
                stats = monitor.get_stats()
                logger.info(f"Parsing completed:")
                logger.info(f"  - Processed {len(df)} log entries")
                logger.info(f"  - Execution time: {stats.execution_time:.2f}s")
                logger.info(f"  - Peak memory: {stats.peak_memory_mb:.1f}MB")
                logger.info(f"  - Throughput: {len(df)/max(stats.execution_time, 0.001):.0f} entries/sec")
                
                return df
                
            except IOError as e:
                raise LogParsingError(f"Error reading file {filepath}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error parsing {filepath}: {e}")
                raise LogParsingError(f"Failed to parse {filepath}: {e}")
    
    def _parse_small_file(self, filepath: Path) -> pd.DataFrame:
        """Parse small files using traditional method."""
        logger.debug("Using traditional parsing for small file")
        
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
                            
        except Exception as e:
            raise LogParsingError(f"Error reading file {filepath}: {e}")
        
        if records:
            df = pd.DataFrame(records)
            logger.debug(f"Successfully parsed {len(df)} log entries")
        else:
            df = pd.DataFrame(columns=["timestamp", "level", "message"])
            logger.warning("No log entries matched any patterns")
        
        return df
    
    def _parse_large_file(self, filepath: Path, file_size_mb: float, available_memory_mb: float) -> pd.DataFrame:
        """Parse large files using batch processing."""
        logger.debug("Using batch processing for large file")
        
        # Calculate optimal chunk size
        optimal_chunk_size = get_optimal_chunk_size(file_size_mb, available_memory_mb)
        chunk_size = min(self.chunk_size, optimal_chunk_size)
        
        logger.debug(f"Using chunk size: {chunk_size}")
        
        # Initialize batch processor
        batch_processor = BatchProcessor(
            chunk_size=chunk_size,
            max_workers=self.max_workers
        )
        
        # Process file in batches
        all_dataframes = []
        
        for batch_df in batch_processor.process_file_in_batches(
            filepath=filepath,
            processor_func=self._process_batch
        ):
            if not batch_df.empty:
                all_dataframes.append(batch_df)
        
        # Combine all batches
        if all_dataframes:
            df = pd.concat(all_dataframes, ignore_index=True)
            logger.debug(f"Combined {len(all_dataframes)} batches into {len(df)} total entries")
        else:
            df = pd.DataFrame(columns=["timestamp", "level", "message"])
            logger.warning("No log entries matched any patterns in any batch")
        
        return df
    
    def _process_batch(self, lines: List[str]) -> pd.DataFrame:
        """Process a batch of log lines."""
        records = []
        
        for line_num, line in enumerate(lines, 1):
            if line.strip():
                parsed_record = self._parse_line(line.strip(), line_num)
                if parsed_record:
                    records.append(parsed_record)
        
        if records:
            return pd.DataFrame(records)
        else:
            return pd.DataFrame(columns=["timestamp", "level", "message"])

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
            try:
                match = pattern.search(line)
                if match:
                    record = match.groupdict()
                    record["line_number"] = line_num
                    
                    # Ensure required fields exist
                    if "message" not in record:
                        record["message"] = line
                    if "timestamp" not in record:
                        record["timestamp"] = None
                    if "level" not in record:
                        record["level"] = "UNKNOWN"
                    
                    return record
                    
            except Exception as e:
                logger.debug(f"Pattern matching error on line {line_num}: {e}")
                continue
        
        # If no pattern matches, create a basic record
        return {
            "timestamp": None,
            "level": "UNPARSED",
            "message": line,
            "line_number": line_num
        }

    def get_supported_fields(self) -> List[str]:
        """
        Get list of fields that can be extracted by current patterns.
        
        Returns:
            List of field names that patterns can extract
        """
        fields = set()
        for pattern in self.patterns:
            fields.update(pattern.groupindex.keys())
        
        # Add standard fields
        fields.update(["line_number", "message", "timestamp", "level"])
        
        return sorted(list(fields))

    def parse_log_lines(self, lines: List[str]) -> pd.DataFrame:
        """
        Parse a list of log lines into a structured DataFrame.
        
        Args:
            lines: List of log lines to parse
            
        Returns:
            DataFrame containing parsed log entries
        """
        records = []
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            parsed_record = self._parse_line(line, line_num)
            if parsed_record:
                records.append(parsed_record)
        
        if records:
            df = pd.DataFrame(records)
            df = MemoryProfiler.optimize_dataframe(df)
            logger.debug(f"Successfully parsed {len(df)} log entries from {len(lines)} lines")
        else:
            df = pd.DataFrame(columns=["timestamp", "level", "message", "line_number"])
            logger.warning("No log entries matched any patterns")
        
        return df
