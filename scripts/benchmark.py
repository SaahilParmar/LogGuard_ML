#!/usr/bin/env python3
"""
Performance Benchmarking Script for LogGuard ML

This script benchmarks various optimization improvements and generates
a comprehensive performance report comparing different configurations.

Usage:
    python scripts/benchmark.py [--iterations N] [--file-sizes SIZES]
"""

import argparse
import time
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple
import tempfile
import json
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from logguard_ml.core.log_parser import LogParser
from logguard_ml.core.advanced_ml import AdvancedAnomalyDetector
from logguard_ml.core.performance import PerformanceMonitor, MemoryProfiler
from logguard_ml.reports.report_generator import generate_html_report
import pandas as pd
import numpy as np


class BenchmarkSuite:
    """Comprehensive benchmarking suite for LogGuard ML."""
    
    def __init__(self, iterations: int = 3):
        """
        Initialize benchmark suite.
        
        Args:
            iterations: Number of iterations for each benchmark
        """
        self.iterations = iterations
        self.results = {}
        
    def generate_test_data(self, num_entries: int) -> str:
        """Generate synthetic log data for testing."""
        levels = ["INFO", "WARN", "ERROR", "DEBUG"]
        messages = [
            "User login successful",
            "Database connection established",
            "API request processed",
            "File uploaded successfully",
            "Cache cleared",
            "Configuration updated",
            "Service started",
            "Task completed",
            "Error processing request",
            "Connection timeout",
            "Invalid parameter",
            "Authentication failed",
            "Memory usage high",
            "Critical system error"
        ]
        
        lines = []
        for i in range(num_entries):
            timestamp = f"2024-01-01 {i//3600:02d}:{(i%3600)//60:02d}:{i%60:02d}"
            level = np.random.choice(levels, p=[0.6, 0.2, 0.1, 0.1])
            message = np.random.choice(messages)
            
            lines.append(f"{timestamp} {level} {message} (entry {i})")
        
        return "\n".join(lines)
    
    def benchmark_parsing(self, config: Dict, file_sizes: List[int]) -> Dict:
        """Benchmark log parsing performance."""
        print("🔍 Benchmarking log parsing...")
        
        results = {}
        
        for size in file_sizes:
            print(f"  Testing with {size:,} log entries...")
            
            # Generate test data
            test_data = self.generate_test_data(size)
            
            # Traditional parsing
            traditional_times = []
            traditional_memory = []
            
            for i in range(self.iterations):
                with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
                    f.write(test_data)
                    temp_path = f.name
                
                try:
                    config_traditional = config.copy()
                    config_traditional['use_parallel_parsing'] = False
                    
                    with PerformanceMonitor() as monitor:
                        parser = LogParser(config_traditional)
                        df = parser.parse_log_file(temp_path)
                    
                    stats = monitor.get_stats()
                    traditional_times.append(stats.execution_time)
                    traditional_memory.append(stats.peak_memory_mb)
                    
                finally:
                    os.unlink(temp_path)
            
            # Parallel parsing
            parallel_times = []
            parallel_memory = []
            
            for i in range(self.iterations):
                with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
                    f.write(test_data)
                    temp_path = f.name
                
                try:
                    config_parallel = config.copy()
                    config_parallel['use_parallel_parsing'] = True
                    config_parallel['chunk_size'] = min(1000, size // 4)
                    
                    with PerformanceMonitor() as monitor:
                        parser = LogParser(config_parallel)
                        df = parser.parse_log_file(temp_path)
                    
                    stats = monitor.get_stats()
                    parallel_times.append(stats.execution_time)
                    parallel_memory.append(stats.peak_memory_mb)
                    
                finally:
                    os.unlink(temp_path)
            
            # Store results
            results[size] = {
                'traditional': {
                    'avg_time': np.mean(traditional_times),
                    'std_time': np.std(traditional_times),
                    'avg_memory': np.mean(traditional_memory),
                    'throughput': size / np.mean(traditional_times)
                },
                'parallel': {
                    'avg_time': np.mean(parallel_times),
                    'std_time': np.std(parallel_times),
                    'avg_memory': np.mean(parallel_memory),
                    'throughput': size / np.mean(parallel_times)
                },
                'improvement': {
                    'time_speedup': np.mean(traditional_times) / np.mean(parallel_times),
                    'memory_ratio': np.mean(parallel_memory) / np.mean(traditional_memory)
                }
            }
            
            print(f"    Traditional: {np.mean(traditional_times):.2f}s ± {np.std(traditional_times):.2f}s")
            print(f"    Parallel: {np.mean(parallel_times):.2f}s ± {np.std(parallel_times):.2f}s")
            print(f"    Speedup: {results[size]['improvement']['time_speedup']:.2f}x")
        
        return results
    
    def benchmark_ml(self, config: Dict, data_sizes: List[int]) -> Dict:
        """Benchmark ML anomaly detection performance."""
        print("🧠 Benchmarking ML anomaly detection...")
        
        results = {}
        
        for size in data_sizes:
            print(f"  Testing with {size:,} log entries...")
            
            # Generate DataFrame
            test_data = self.generate_test_data(size)
            lines = test_data.split('\n')
            
            # Parse into DataFrame
            config_parse = config.copy()
            config_parse['use_parallel_parsing'] = False
            parser = LogParser(config_parse)
            
            records = []
            for line_num, line in enumerate(lines):
                parsed = parser._parse_line(line.strip(), line_num)
                if parsed:
                    records.append(parsed)
            
            df = pd.DataFrame(records)
            
            # Test different algorithms
            algorithms = ['isolation_forest', 'one_class_svm', 'ensemble']
            
            for algorithm in algorithms:
                print(f"    Testing {algorithm}...")
                
                times = []
                memory_usage = []
                
                for i in range(self.iterations):
                    config_ml = config.copy()
                    config_ml['ml_model']['algorithm'] = algorithm
                    
                    with PerformanceMonitor() as monitor:
                        detector = AdvancedAnomalyDetector(config_ml)
                        df_with_anomalies = detector.detect_anomalies(df.copy())
                    
                    stats = monitor.get_stats()
                    times.append(stats.execution_time)
                    memory_usage.append(stats.peak_memory_mb)
                
                results[f"{size}_{algorithm}"] = {
                    'avg_time': np.mean(times),
                    'std_time': np.std(times),
                    'avg_memory': np.mean(memory_usage),
                    'throughput': size / np.mean(times),
                    'anomalies_detected': int(df_with_anomalies['is_anomaly'].sum())
                }
                
                print(f"      Time: {np.mean(times):.2f}s ± {np.std(times):.2f}s")
                print(f"      Memory: {np.mean(memory_usage):.1f}MB")
                print(f"      Anomalies: {results[f'{size}_{algorithm}']['anomalies_detected']}")
        
        return results
    
    def benchmark_memory_optimization(self, config: Dict, data_sizes: List[int]) -> Dict:
        """Benchmark memory optimization features."""
        print("💾 Benchmarking memory optimization...")
        
        results = {}
        
        for size in data_sizes:
            print(f"  Testing with {size:,} log entries...")
            
            # Generate test DataFrame
            test_data = self.generate_test_data(size)
            lines = test_data.split('\n')
            
            config_parse = config.copy()
            parser = LogParser(config_parse)
            
            records = []
            for line_num, line in enumerate(lines):
                parsed = parser._parse_line(line.strip(), line_num)
                if parsed:
                    records.append(parsed)
            
            # Test without optimization
            df_original = pd.DataFrame(records)
            original_memory = df_original.memory_usage(deep=True).sum() / (1024 * 1024)
            
            # Test with optimization
            df_optimized = MemoryProfiler.optimize_dataframe(df_original.copy())
            optimized_memory = df_optimized.memory_usage(deep=True).sum() / (1024 * 1024)
            
            reduction = (original_memory - optimized_memory) / original_memory * 100
            
            results[size] = {
                'original_memory_mb': original_memory,
                'optimized_memory_mb': optimized_memory,
                'reduction_percent': reduction,
                'compression_ratio': original_memory / optimized_memory
            }
            
            print(f"    Original: {original_memory:.1f}MB")
            print(f"    Optimized: {optimized_memory:.1f}MB")
            print(f"    Reduction: {reduction:.1f}%")
        
        return results
    
    def benchmark_report_generation(self, config: Dict, data_sizes: List[int]) -> Dict:
        """Benchmark report generation performance."""
        print("📊 Benchmarking report generation...")
        
        results = {}
        
        for size in data_sizes:
            print(f"  Testing with {size:,} log entries...")
            
            # Generate test data with anomalies
            test_data = self.generate_test_data(size)
            lines = test_data.split('\n')
            
            # Create DataFrame with anomalies
            parser = LogParser(config)
            records = []
            for line_num, line in enumerate(lines):
                parsed = parser._parse_line(line.strip(), line_num)
                if parsed:
                    records.append(parsed)
            
            df = pd.DataFrame(records)
            detector = AdvancedAnomalyDetector(config)
            df_with_anomalies = detector.detect_anomalies(df)
            
            # Test with caching
            cached_times = []
            for i in range(self.iterations):
                with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
                    temp_output = f.name
                
                try:
                    with PerformanceMonitor() as monitor:
                        generate_html_report(df_with_anomalies, temp_output, use_cache=True)
                    
                    stats = monitor.get_stats()
                    cached_times.append(stats.execution_time)
                    
                finally:
                    os.unlink(temp_output)
            
            # Test without caching
            no_cache_times = []
            for i in range(self.iterations):
                with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
                    temp_output = f.name
                
                try:
                    with PerformanceMonitor() as monitor:
                        generate_html_report(df_with_anomalies, temp_output, use_cache=False)
                    
                    stats = monitor.get_stats()
                    no_cache_times.append(stats.execution_time)
                    
                finally:
                    os.unlink(temp_output)
            
            results[size] = {
                'cached': {
                    'avg_time': np.mean(cached_times),
                    'std_time': np.std(cached_times)
                },
                'no_cache': {
                    'avg_time': np.mean(no_cache_times),
                    'std_time': np.std(no_cache_times)
                },
                'cache_speedup': np.mean(no_cache_times) / np.mean(cached_times)
            }
            
            print(f"    With cache: {np.mean(cached_times):.2f}s ± {np.std(cached_times):.2f}s")
            print(f"    No cache: {np.mean(no_cache_times):.2f}s ± {np.std(no_cache_times):.2f}s")
            print(f"    Speedup: {results[size]['cache_speedup']:.2f}x")
        
        return results
    
    def run_full_benchmark(self, config: Dict, file_sizes: List[int]) -> Dict:
        """Run comprehensive benchmark suite."""
        print("🚀 Starting LogGuard ML Performance Benchmark")
        print(f"Iterations per test: {self.iterations}")
        print(f"File sizes: {file_sizes}")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run all benchmarks
        self.results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'iterations': self.iterations,
                'file_sizes': file_sizes,
                'config': config
            },
            'parsing': self.benchmark_parsing(config, file_sizes),
            'ml': self.benchmark_ml(config, file_sizes),
            'memory': self.benchmark_memory_optimization(config, file_sizes),
            'reports': self.benchmark_report_generation(config, file_sizes)
        }
        
        total_time = time.time() - start_time
        self.results['metadata']['total_benchmark_time'] = total_time
        
        print("=" * 60)
        print(f"✅ Benchmark completed in {total_time:.1f}s")
        
        return self.results
    
    def generate_report(self, output_file: str):
        """Generate a comprehensive benchmark report."""
        print(f"📝 Generating benchmark report: {output_file}")
        
        # Save JSON results
        json_output = output_file.replace('.html', '_results.json')
        with open(json_output, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Generate HTML report
        html_content = self._create_html_report()
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        print(f"📊 Report saved to: {output_file}")
        print(f"📄 Raw data saved to: {json_output}")
    
    def _create_html_report(self) -> str:
        """Create HTML benchmark report."""
        parsing_section = self._create_parsing_section()
        ml_section = self._create_ml_section()
        memory_section = self._create_memory_section()
        reports_section = self._create_reports_section()
        
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>LogGuard ML Performance Benchmark Report</title>
    <meta charset="utf-8">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ text-align: center; margin-bottom: 40px; }}
        .section {{ margin: 30px 0; }}
        .metric {{ background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .speedup {{ color: green; font-weight: bold; }}
        .summary {{ background: #e7f3ff; padding: 20px; border-radius: 10px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>LogGuard ML Performance Benchmark Report</h1>
        <p>Generated: {self.results['metadata']['timestamp']}</p>
        <p>Iterations: {self.results['metadata']['iterations']}</p>
        <p>Total benchmark time: {self.results['metadata'].get('total_benchmark_time', 0):.1f}s</p>
    </div>
    
    <div class="summary">
        <h2>Executive Summary</h2>
        {self._create_summary()}
    </div>
    
    {parsing_section}
    {ml_section}
    {memory_section}
    {reports_section}
    
    <footer style="margin-top: 50px; text-align: center; color: #666;">
        <p>LogGuard ML Performance Benchmark - Generated by LogGuard ML v0.1.0</p>
    </footer>
</body>
</html>
        """
    
    def _create_summary(self) -> str:
        """Create executive summary."""
        # Calculate key metrics
        parsing_results = self.results.get('parsing', {})
        max_speedup = 0
        avg_memory_reduction = 0
        
        if parsing_results:
            speedups = [data['improvement']['time_speedup'] for data in parsing_results.values()]
            max_speedup = max(speedups) if speedups else 0
        
        memory_results = self.results.get('memory', {})
        if memory_results:
            reductions = [data['reduction_percent'] for data in memory_results.values()]
            avg_memory_reduction = np.mean(reductions) if reductions else 0
        
        return f"""
        <ul>
            <li><strong>Maximum Parsing Speedup:</strong> {max_speedup:.2f}x with parallel processing</li>
            <li><strong>Average Memory Reduction:</strong> {avg_memory_reduction:.1f}% with optimization</li>
            <li><strong>ML Algorithms Tested:</strong> Isolation Forest, One-Class SVM, Ensemble</li>
            <li><strong>Caching Benefits:</strong> Significant improvement in report generation</li>
        </ul>
        """
    
    def _create_parsing_section(self) -> str:
        """Create parsing performance section."""
        if 'parsing' not in self.results:
            return ""
        
        rows = []
        for size, data in self.results['parsing'].items():
            rows.append(f"""
            <tr>
                <td>{size:,}</td>
                <td>{data['traditional']['avg_time']:.2f}s</td>
                <td>{data['parallel']['avg_time']:.2f}s</td>
                <td class="speedup">{data['improvement']['time_speedup']:.2f}x</td>
                <td>{data['traditional']['throughput']:.0f}</td>
                <td>{data['parallel']['throughput']:.0f}</td>
            </tr>
            """)
        
        return f"""
        <div class="section">
            <h2>🔍 Log Parsing Performance</h2>
            <table>
                <tr>
                    <th>Entries</th>
                    <th>Traditional Time</th>
                    <th>Parallel Time</th>
                    <th>Speedup</th>
                    <th>Traditional Throughput</th>
                    <th>Parallel Throughput</th>
                </tr>
                {''.join(rows)}
            </table>
        </div>
        """
    
    def _create_ml_section(self) -> str:
        """Create ML performance section."""
        if 'ml' not in self.results:
            return ""
        
        algorithms = ['isolation_forest', 'one_class_svm', 'ensemble']
        rows = []
        
        for key, data in self.results['ml'].items():
            if '_' in key:
                size, algorithm = key.rsplit('_', 1)
                rows.append(f"""
                <tr>
                    <td>{size}</td>
                    <td>{algorithm.replace('_', ' ').title()}</td>
                    <td>{data['avg_time']:.2f}s</td>
                    <td>{data['avg_memory']:.1f}MB</td>
                    <td>{data['throughput']:.0f}</td>
                    <td>{data['anomalies_detected']}</td>
                </tr>
                """)
        
        return f"""
        <div class="section">
            <h2>🧠 ML Anomaly Detection Performance</h2>
            <table>
                <tr>
                    <th>Entries</th>
                    <th>Algorithm</th>
                    <th>Time</th>
                    <th>Memory</th>
                    <th>Throughput</th>
                    <th>Anomalies</th>
                </tr>
                {''.join(rows)}
            </table>
        </div>
        """
    
    def _create_memory_section(self) -> str:
        """Create memory optimization section."""
        if 'memory' not in self.results:
            return ""
        
        rows = []
        for size, data in self.results['memory'].items():
            rows.append(f"""
            <tr>
                <td>{size:,}</td>
                <td>{data['original_memory_mb']:.1f}MB</td>
                <td>{data['optimized_memory_mb']:.1f}MB</td>
                <td class="speedup">{data['reduction_percent']:.1f}%</td>
                <td>{data['compression_ratio']:.2f}x</td>
            </tr>
            """)
        
        return f"""
        <div class="section">
            <h2>💾 Memory Optimization Performance</h2>
            <table>
                <tr>
                    <th>Entries</th>
                    <th>Original Memory</th>
                    <th>Optimized Memory</th>
                    <th>Reduction</th>
                    <th>Compression Ratio</th>
                </tr>
                {''.join(rows)}
            </table>
        </div>
        """
    
    def _create_reports_section(self) -> str:
        """Create report generation section."""
        if 'reports' not in self.results:
            return ""
        
        rows = []
        for size, data in self.results['reports'].items():
            rows.append(f"""
            <tr>
                <td>{size:,}</td>
                <td>{data['no_cache']['avg_time']:.2f}s</td>
                <td>{data['cached']['avg_time']:.2f}s</td>
                <td class="speedup">{data['cache_speedup']:.2f}x</td>
            </tr>
            """)
        
        return f"""
        <div class="section">
            <h2>📊 Report Generation Performance</h2>
            <table>
                <tr>
                    <th>Entries</th>
                    <th>No Cache</th>
                    <th>With Cache</th>
                    <th>Speedup</th>
                </tr>
                {''.join(rows)}
            </table>
        </div>
        """


def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(description="LogGuard ML Performance Benchmark")
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations per test")
    parser.add_argument("--file-sizes", nargs="+", type=int, default=[1000, 5000, 10000], 
                       help="File sizes to test (number of log entries)")
    parser.add_argument("--output", default="benchmark_report.html", help="Output report file")
    
    args = parser.parse_args()
    
    # Default configuration
    config = {
        "log_patterns": [
            {"pattern": r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (?P<level>ERROR|WARN|INFO|DEBUG) (?P<message>.+)"}
        ],
        "ml_model": {
            "algorithm": "isolation_forest",
            "contamination": 0.05,
            "random_state": 42,
            "max_samples": "auto"
        },
        "use_parallel_parsing": True,
        "chunk_size": 1000
    }
    
    # Run benchmark
    benchmark = BenchmarkSuite(iterations=args.iterations)
    results = benchmark.run_full_benchmark(config, args.file_sizes)
    
    # Generate report
    benchmark.generate_report(args.output)
    
    print("\n🎯 Benchmark Summary:")
    print(f"  - Report: {args.output}")
    print(f"  - Raw data: {args.output.replace('.html', '_results.json')}")
    print("  - Review the report for detailed performance insights!")


if __name__ == "__main__":
    main()
