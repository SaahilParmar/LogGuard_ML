"""
Performance Monitoring and Optimization Module for LogGuard ML

This module provides comprehensive performance monitoring, memory profiling,
and optimization utilities for the LogGuard ML framework.

Classes:
    PerformanceMonitor: Main class for monitoring system performance
    MemoryProfiler: Memory usage tracking and optimization
    BatchProcessor: Optimized batch processing for large log files

Example:
    >>> from logguard_ml.core.performance import PerformanceMonitor
    >>> with PerformanceMonitor() as monitor:
    >>>     # Your code here
    >>>     pass
    >>> print(monitor.get_stats())
"""

import gc
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import psutil

logger = logging.getLogger(__name__)


@dataclass
class PerformanceStats:
    """Performance statistics container."""

    execution_time: float
    peak_memory_mb: float
    cpu_percent: float
    memory_percent: float
    io_read_bytes: int
    io_write_bytes: int
    gc_collections: int


class PerformanceMonitor:
    """
    Advanced performance monitoring with real-time metrics collection.

    Tracks execution time, memory usage, CPU utilization, and I/O operations
    to identify performance bottlenecks and optimization opportunities.
    """

    def __init__(self, sample_interval: float = 0.1):
        """
        Initialize performance monitor.

        Args:
            sample_interval: How often to sample system metrics (seconds)
        """
        self.sample_interval = sample_interval
        self.start_time = None
        self.end_time = None
        self.peak_memory = 0
        self.samples = []
        self._monitoring = False
        self._monitor_thread = None
        self.process = psutil.Process()

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

    def start(self):
        """Start performance monitoring."""
        logger.debug("Starting performance monitoring")
        self.start_time = time.time()
        self.peak_memory = 0
        self.samples = []
        self._monitoring = True

        # Start monitoring thread
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop(self):
        """Stop performance monitoring."""
        self.end_time = time.time()
        self._monitoring = False

        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)

        logger.debug(
            f"Performance monitoring stopped after {self.get_execution_time():.2f}s"
        )

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._monitoring:
            try:
                # Collect system metrics
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                cpu_percent = self.process.cpu_percent()

                self.peak_memory = max(self.peak_memory, memory_mb)

                sample = {
                    "timestamp": time.time(),
                    "memory_mb": memory_mb,
                    "cpu_percent": cpu_percent,
                }
                self.samples.append(sample)

                time.sleep(self.sample_interval)

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break

    def get_execution_time(self) -> float:
        """Get total execution time."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time

    def get_stats(self) -> PerformanceStats:
        """Get comprehensive performance statistics."""
        io_counters = self.process.io_counters()
        memory_info = self.process.memory_info()

        return PerformanceStats(
            execution_time=self.get_execution_time(),
            peak_memory_mb=self.peak_memory,
            cpu_percent=self.process.cpu_percent(),
            memory_percent=self.process.memory_percent(),
            io_read_bytes=io_counters.read_bytes,
            io_write_bytes=io_counters.write_bytes,
            gc_collections=gc.get_count()[0],
        )

    def log_stats(self):
        """Log performance statistics."""
        stats = self.get_stats()
        logger.info(f"Performance Stats:")
        logger.info(f"  Execution Time: {stats.execution_time:.2f}s")
        logger.info(f"  Peak Memory: {stats.peak_memory_mb:.1f} MB")
        logger.info(f"  CPU Usage: {stats.cpu_percent:.1f}%")
        logger.info(f"  Memory Usage: {stats.memory_percent:.1f}%")
        logger.info(f"  I/O Read: {stats.io_read_bytes / (1024*1024):.1f} MB")
        logger.info(f"  I/O Write: {stats.io_write_bytes / (1024*1024):.1f} MB")


class MemoryProfiler:
    """Memory usage profiler with optimization recommendations."""

    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "rss_mb": memory_info.rss / (1024 * 1024),
            "vms_mb": memory_info.vms / (1024 * 1024),
            "percent": process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / (1024 * 1024),
        }

    @staticmethod
    def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage by downcasting numeric types.

        Args:
            df: Input DataFrame

        Returns:
            Memory-optimized DataFrame
        """
        if df.empty:
            return df

        logger.debug(f"Optimizing DataFrame memory usage (shape: {df.shape})")

        original_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)

        # Optimize numeric columns
        for col in df.select_dtypes(include=["int64"]).columns:
            if (
                df[col].min() >= np.iinfo(np.int32).min
                and df[col].max() <= np.iinfo(np.int32).max
            ):
                df[col] = df[col].astype(np.int32)
            elif (
                df[col].min() >= np.iinfo(np.int16).min
                and df[col].max() <= np.iinfo(np.int16).max
            ):
                df[col] = df[col].astype(np.int16)
            elif (
                df[col].min() >= np.iinfo(np.int8).min
                and df[col].max() <= np.iinfo(np.int8).max
            ):
                df[col] = df[col].astype(np.int8)

        for col in df.select_dtypes(include=["float64"]).columns:
            if (
                df[col].min() >= np.finfo(np.float32).min
                and df[col].max() <= np.finfo(np.float32).max
            ):
                df[col] = df[col].astype(np.float32)

        # Optimize string columns to categorical where beneficial
        for col in df.select_dtypes(include=["object"]).columns:
            if df[col].dtype == "object":
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.5:  # Less than 50% unique values
                    df[col] = df[col].astype("category")

        optimized_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)
        reduction = (original_memory - optimized_memory) / original_memory * 100

        logger.debug(
            f"Memory optimization: {original_memory:.1f}MB -> {optimized_memory:.1f}MB ({reduction:.1f}% reduction)"
        )

        return df

    @staticmethod
    def force_garbage_collection():
        """Force garbage collection to free memory."""
        collected = gc.collect()
        logger.debug(f"Garbage collection freed {collected} objects")
        return collected


class BatchProcessor:
    """
    Optimized batch processor for handling large log files efficiently.

    Processes files in chunks to maintain constant memory usage regardless
    of file size, with parallel processing support for CPU-intensive operations.
    """

    def __init__(self, chunk_size: int = 10000, max_workers: int = None):
        """
        Initialize batch processor.

        Args:
            chunk_size: Number of lines to process per batch
            max_workers: Maximum number of worker threads (None = auto)
        """
        self.chunk_size = chunk_size
        self.max_workers = max_workers or min(4, psutil.cpu_count())
        logger.debug(
            f"BatchProcessor initialized: chunk_size={chunk_size}, max_workers={self.max_workers}"
        )

    def process_file_in_batches(
        self, filepath: Union[str, Path], processor_func, **kwargs
    ) -> Iterator[pd.DataFrame]:
        """
        Process a large file in memory-efficient batches.

        Args:
            filepath: Path to the file to process
            processor_func: Function to apply to each batch
            **kwargs: Additional arguments for processor_func

        Yields:
            Processed DataFrame batches
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        logger.info(f"Processing file in batches: {filepath}")

        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as file:
                batch_lines = []
                batch_num = 0

                for line_num, line in enumerate(file, 1):
                    batch_lines.append(line.strip())

                    if len(batch_lines) >= self.chunk_size:
                        batch_num += 1
                        logger.debug(
                            f"Processing batch {batch_num} (lines {line_num - self.chunk_size + 1}-{line_num})"
                        )

                        # Process batch
                        batch_df = processor_func(batch_lines, **kwargs)
                        if not batch_df.empty:
                            yield MemoryProfiler.optimize_dataframe(batch_df)

                        # Clear batch and force GC
                        batch_lines = []
                        MemoryProfiler.force_garbage_collection()

                # Process remaining lines
                if batch_lines:
                    batch_num += 1
                    logger.debug(
                        f"Processing final batch {batch_num} ({len(batch_lines)} lines)"
                    )
                    batch_df = processor_func(batch_lines, **kwargs)
                    if not batch_df.empty:
                        yield MemoryProfiler.optimize_dataframe(batch_df)

        except Exception as e:
            logger.error(f"Error processing file in batches: {e}")
            raise

    def parallel_process_batches(
        self, batches: List[pd.DataFrame], processor_func, **kwargs
    ) -> List[pd.DataFrame]:
        """
        Process multiple batches in parallel using ThreadPoolExecutor.

        Args:
            batches: List of DataFrame batches to process
            processor_func: Function to apply to each batch
            **kwargs: Additional arguments for processor_func

        Returns:
            List of processed DataFrames
        """
        if not batches:
            return []

        logger.info(
            f"Processing {len(batches)} batches in parallel with {self.max_workers} workers"
        )

        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_batch = {
                executor.submit(processor_func, batch, **kwargs): i
                for i, batch in enumerate(batches)
            }

            # Collect results in order
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    result = future.result()
                    results.append((batch_idx, result))
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {e}")
                    results.append((batch_idx, pd.DataFrame()))

        # Sort results by batch index to maintain order
        results.sort(key=lambda x: x[0])
        return [result for _, result in results]


@contextmanager
def performance_context(operation_name: str = "operation"):
    """
    Context manager for easy performance monitoring.

    Args:
        operation_name: Name of the operation being monitored

    Example:
        >>> with performance_context("log_parsing"):
        >>>     # Your code here
        >>>     pass
    """
    monitor = PerformanceMonitor()
    logger.info(f"Starting {operation_name}")

    try:
        monitor.start()
        yield monitor
    finally:
        monitor.stop()
        stats = monitor.get_stats()
        logger.info(f"Completed {operation_name} in {stats.execution_time:.2f}s")
        logger.info(f"Peak memory usage: {stats.peak_memory_mb:.1f}MB")


def profile_function(func):
    """
    Decorator for automatic function performance profiling.

    Example:
        >>> @profile_function
        >>> def my_function():
        >>>     # Function code
        >>>     pass
    """

    def wrapper(*args, **kwargs):
        with performance_context(func.__name__) as monitor:
            result = func(*args, **kwargs)
        return result

    return wrapper


# Optimization utility functions
def optimize_pandas_settings():
    """Configure pandas for optimal performance."""
    pd.set_option("mode.copy_on_write", True)  # Reduce memory usage
    pd.set_option("compute.use_bottleneck", True)  # Use optimized functions
    pd.set_option("compute.use_numexpr", True)  # Use fast expression evaluation
    logger.debug("Pandas optimization settings applied")


def get_optimal_chunk_size(file_size_mb: float, available_memory_mb: float) -> int:
    """
    Calculate optimal chunk size based on file size and available memory.

    Args:
        file_size_mb: Size of the file in MB
        available_memory_mb: Available system memory in MB

    Returns:
        Optimal chunk size in number of lines
    """
    # Use 10% of available memory or 100MB, whichever is smaller
    target_memory_mb = min(available_memory_mb * 0.1, 100)

    # Estimate lines per MB (rough estimate: 100 lines per MB for typical logs)
    lines_per_mb = 100

    # Calculate chunk size
    chunk_size = int(target_memory_mb * lines_per_mb)

    # Ensure reasonable bounds
    chunk_size = max(1000, min(chunk_size, 50000))

    logger.debug(f"Calculated optimal chunk size: {chunk_size} lines")
    return chunk_size
