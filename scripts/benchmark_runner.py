#!/usr/bin/env python3
"""
Performance Benchmark Runner for LogGuard ML

This script provides automated performance regression testing to ensure
that code changes don't negatively impact performance.

Features:
- Automated benchmark execution
- Performance regression detection
- Historical performance tracking
- Detailed performance reports
- CI/CD integration support

Usage:
    python scripts/benchmark_runner.py --run-all
    python scripts/benchmark_runner.py --compare baseline.json current.json
    python scripts/benchmark_runner.py --profile --iterations 10
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import statistics
import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from logguard_ml import LogParser, AdvancedAnomalyDetector, generate_html_report
from logguard_ml.core.performance import PerformanceMonitor, MemoryProfiler
import yaml
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Automated performance benchmark runner."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize benchmark runner."""
        self.config_path = config_path or "config/config.yaml"
        self.results_dir = Path("benchmarks/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)
            
        # Create test data
        self.test_data_dir = Path("benchmarks/test_data")
        self.test_data_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_test_data(self, size: str = "medium") -> Path:
        """Generate synthetic log data for benchmarking."""
        sizes = {
            "small": 1000,
            "medium": 10000,
            "large": 50000,
            "xlarge": 100000
        }
        
        num_lines = sizes.get(size, 10000)
        test_file = self.test_data_dir / f"test_log_{size}_{num_lines}.log"
        
        if test_file.exists():
            return test_file
            
        logger.info(f"Generating {size} test data with {num_lines} lines")
        
        # Generate realistic log entries
        levels = ["INFO", "WARN", "ERROR", "DEBUG"]
        messages = [
            "Application started successfully",
            "User authentication completed",
            "Database connection established",
            "Processing request",
            "Cache hit for key",
            "API response sent",
            "Background job completed",
            "Configuration loaded",
            "Memory usage: 85%",
            "Network timeout occurred",
            "Invalid input received",
            "Service unavailable",
            "Critical system error",
            "Disk space low",
            "Performance degradation detected"
        ]
        
        with open(test_file, 'w') as f:
            for i in range(num_lines):
                timestamp = f"2025-07-{14 + i % 30:02d} {10 + i % 14:02d}:{i % 60:02d}:{(i * 7) % 60:02d}"
                level = levels[i % len(levels)]
                message = messages[i % len(messages)]
                
                # Add some anomalous entries (5%)
                if i % 20 == 0:
                    message = f"ANOMALOUS_PATTERN_{i}_UNUSUAL_BEHAVIOR_DETECTED"
                    
                f.write(f"{timestamp} {level} {message}\n")
                
        logger.info(f"Generated test data: {test_file}")
        return test_file
        
    def benchmark_log_parsing(self, test_file: Path, iterations: int = 5) -> Dict:
        """Benchmark log parsing performance."""
        logger.info(f"Benchmarking log parsing on {test_file.name}")
        
        results = []
        parser = LogParser(self.config)
        
        for i in range(iterations):
            with PerformanceMonitor() as monitor:
                df = parser.parse_log_file(str(test_file))
                
            stats = monitor.get_stats()
            results.append({
                "iteration": i + 1,
                "execution_time": stats.execution_time,
                "peak_memory_mb": stats.peak_memory_mb,
                "rows_processed": len(df),
                "processing_rate": len(df) / stats.execution_time
            })
            
        return {
            "benchmark": "log_parsing",
            "test_file": test_file.name,
            "iterations": iterations,
            "results": results,
            "avg_execution_time": statistics.mean([r["execution_time"] for r in results]),
            "avg_memory_mb": statistics.mean([r["peak_memory_mb"] for r in results]),
            "avg_processing_rate": statistics.mean([r["processing_rate"] for r in results]),
            "std_execution_time": statistics.stdev([r["execution_time"] for r in results]) if len(results) > 1 else 0
        }
        
    def benchmark_anomaly_detection(self, test_file: Path, iterations: int = 5) -> Dict:
        """Benchmark ML anomaly detection performance."""
        logger.info(f"Benchmarking anomaly detection on {test_file.name}")
        
        # First parse the logs
        parser = LogParser(self.config)
        df = parser.parse_log_file(str(test_file))
        
        results = []
        detector = AdvancedAnomalyDetector(self.config)
        
        for i in range(iterations):
            with PerformanceMonitor() as monitor:
                anomaly_df = detector.detect_anomalies(df.copy())
                
            stats = monitor.get_stats()
            anomaly_count = anomaly_df['is_anomaly'].sum() if 'is_anomaly' in anomaly_df.columns else 0
            
            results.append({
                "iteration": i + 1,
                "execution_time": stats.execution_time,
                "peak_memory_mb": stats.peak_memory_mb,
                "rows_processed": len(df),
                "anomalies_detected": anomaly_count,
                "processing_rate": len(df) / stats.execution_time
            })
            
        return {
            "benchmark": "anomaly_detection",
            "test_file": test_file.name,
            "iterations": iterations,
            "results": results,
            "avg_execution_time": statistics.mean([r["execution_time"] for r in results]),
            "avg_memory_mb": statistics.mean([r["peak_memory_mb"] for r in results]),
            "avg_processing_rate": statistics.mean([r["processing_rate"] for r in results]),
            "avg_anomalies": statistics.mean([r["anomalies_detected"] for r in results]),
            "std_execution_time": statistics.stdev([r["execution_time"] for r in results]) if len(results) > 1 else 0
        }
        
    def benchmark_report_generation(self, test_file: Path, iterations: int = 3) -> Dict:
        """Benchmark HTML report generation performance."""
        logger.info(f"Benchmarking report generation on {test_file.name}")
        
        # Prepare data
        parser = LogParser(self.config)
        df = parser.parse_log_file(str(test_file))
        detector = AdvancedAnomalyDetector(self.config)
        anomaly_df = detector.detect_anomalies(df)
        
        results = []
        
        for i in range(iterations):
            output_file = self.results_dir / f"benchmark_report_{i}.html"
            
            with PerformanceMonitor() as monitor:
                generate_html_report(
                    anomaly_df, 
                    str(output_file),
                    title=f"Benchmark Report {i+1}"
                )
                
            stats = monitor.get_stats()
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            
            results.append({
                "iteration": i + 1,
                "execution_time": stats.execution_time,
                "peak_memory_mb": stats.peak_memory_mb,
                "report_size_mb": file_size_mb,
                "rows_processed": len(anomaly_df)
            })
            
            # Clean up
            output_file.unlink(missing_ok=True)
            
        return {
            "benchmark": "report_generation",
            "test_file": test_file.name,
            "iterations": iterations,
            "results": results,
            "avg_execution_time": statistics.mean([r["execution_time"] for r in results]),
            "avg_memory_mb": statistics.mean([r["peak_memory_mb"] for r in results]),
            "avg_report_size_mb": statistics.mean([r["report_size_mb"] for r in results]),
            "std_execution_time": statistics.stdev([r["execution_time"] for r in results]) if len(results) > 1 else 0
        }
        
    def run_full_benchmark_suite(self, iterations: int = 5) -> Dict:
        """Run complete benchmark suite."""
        logger.info("Starting full benchmark suite")
        
        # Test different data sizes
        test_sizes = ["small", "medium", "large"]
        all_results = {
            "timestamp": datetime.now().isoformat(),
            "iterations": iterations,
            "system_info": self._get_system_info(),
            "benchmarks": {}
        }
        
        for size in test_sizes:
            logger.info(f"Running benchmarks for {size} dataset")
            test_file = self.generate_test_data(size)
            
            all_results["benchmarks"][size] = {
                "log_parsing": self.benchmark_log_parsing(test_file, iterations),
                "anomaly_detection": self.benchmark_anomaly_detection(test_file, iterations),
                "report_generation": self.benchmark_report_generation(test_file, max(1, iterations // 2))
            }
            
        return all_results
        
    def _get_logguard_version(self) -> str:
        """Get LogGuard ML version dynamically."""
        try:
            from logguard_ml.__version__ import __version__
            return __version__
        except ImportError:
            return "unknown"

    def _get_system_info(self) -> Dict:
        """Get system information for benchmark context."""
        import platform
        import psutil
        
        return {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "logguard_version": self._get_logguard_version()
        }
        
    def save_results(self, results: Dict, filename: Optional[str] = None) -> Path:
        """Save benchmark results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
            
        results_file = self.results_dir / filename
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            return obj
        
        results = convert_numpy_types(results)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Benchmark results saved to: {results_file}")
        return results_file
        
    def compare_results(self, baseline_file: Path, current_file: Path) -> Dict:
        """Compare two benchmark result files for regression detection."""
        logger.info(f"Comparing {baseline_file.name} vs {current_file.name}")
        
        with open(baseline_file) as f:
            baseline = json.load(f)
        with open(current_file) as f:
            current = json.load(f)
            
        comparison = {
            "baseline_file": baseline_file.name,
            "current_file": current_file.name,
            "comparison_timestamp": datetime.now().isoformat(),
            "regressions": [],
            "improvements": [],
            "summary": {}
        }
        
        # Compare each benchmark category
        for size in baseline.get("benchmarks", {}):
            if size not in current.get("benchmarks", {}):
                continue
                
            baseline_size = baseline["benchmarks"][size]
            current_size = current["benchmarks"][size]
            
            for benchmark_type in baseline_size:
                if benchmark_type not in current_size:
                    continue
                    
                baseline_bench = baseline_size[benchmark_type]
                current_bench = current_size[benchmark_type]
                
                # Compare execution time
                baseline_time = baseline_bench["avg_execution_time"]
                current_time = current_bench["avg_execution_time"]
                time_change = ((current_time - baseline_time) / baseline_time) * 100
                
                # Compare memory usage
                baseline_memory = baseline_bench["avg_memory_mb"]
                current_memory = current_bench["avg_memory_mb"]
                memory_change = ((current_memory - baseline_memory) / baseline_memory) * 100
                
                bench_comparison = {
                    "benchmark": f"{size}_{benchmark_type}",
                    "execution_time_change_percent": round(time_change, 2),
                    "memory_change_percent": round(memory_change, 2),
                    "baseline_time": baseline_time,
                    "current_time": current_time,
                    "baseline_memory": baseline_memory,
                    "current_memory": current_memory
                }
                
                # Flag regressions (>10% slower or >20% more memory)
                if time_change > 10 or memory_change > 20:
                    comparison["regressions"].append(bench_comparison)
                elif time_change < -5 or memory_change < -10:
                    comparison["improvements"].append(bench_comparison)
                    
        comparison["summary"] = {
            "total_regressions": len(comparison["regressions"]),
            "total_improvements": len(comparison["improvements"]),
            "has_significant_regressions": len(comparison["regressions"]) > 0
        }
        
        return comparison
        
    def generate_benchmark_report(self, results: Dict) -> str:
        """Generate a human-readable benchmark report."""
        report = []
        report.append("# LogGuard ML Performance Benchmark Report")
        report.append(f"Generated: {results['timestamp']}")
        report.append(f"Iterations: {results['iterations']}")
        report.append("")
        
        # System info
        system_info = results["system_info"]
        report.append("## System Information")
        report.append(f"- Python: {system_info['python_version']}")
        report.append(f"- Platform: {system_info['platform']}")
        report.append(f"- CPU Cores: {system_info['cpu_count']}")
        report.append(f"- Memory: {system_info['memory_gb']} GB")
        report.append("")
        
        # Results summary
        report.append("## Performance Results")
        
        for size, benchmarks in results["benchmarks"].items():
            report.append(f"### {size.upper()} Dataset")
            
            for bench_name, bench_data in benchmarks.items():
                report.append(f"#### {bench_name.replace('_', ' ').title()}")
                report.append(f"- Average Execution Time: {bench_data['avg_execution_time']:.3f}s")
                report.append(f"- Average Memory Usage: {bench_data['avg_memory_mb']:.1f} MB")
                
                if "avg_processing_rate" in bench_data:
                    report.append(f"- Processing Rate: {bench_data['avg_processing_rate']:.0f} logs/sec")
                if "avg_anomalies" in bench_data:
                    report.append(f"- Average Anomalies Detected: {bench_data['avg_anomalies']:.1f}")
                    
                report.append("")
                
        return "\n".join(report)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="LogGuard ML Performance Benchmark Runner")
    
    parser.add_argument("--run-all", action="store_true",
                       help="Run complete benchmark suite")
    parser.add_argument("--iterations", type=int, default=5,
                       help="Number of iterations per benchmark")
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE", "CURRENT"),
                       help="Compare two benchmark result files")
    parser.add_argument("--generate-data", choices=["small", "medium", "large", "xlarge"],
                       help="Generate test data of specified size")
    parser.add_argument("--config", default="config/config.yaml",
                       help="Configuration file path")
    parser.add_argument("--output", help="Output filename for results")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    runner = BenchmarkRunner(args.config)
    
    if args.generate_data:
        test_file = runner.generate_test_data(args.generate_data)
        print(f"Generated test data: {test_file}")
        return
        
    if args.compare:
        baseline_file = Path(args.compare[0])
        current_file = Path(args.compare[1])
        
        if not baseline_file.exists() or not current_file.exists():
            print("Error: Comparison files must exist")
            return 1
            
        comparison = runner.compare_results(baseline_file, current_file)
        
        print("# Benchmark Comparison Results")
        print(f"Baseline: {comparison['baseline_file']}")
        print(f"Current: {comparison['current_file']}")
        print()
        
        if comparison["summary"]["has_significant_regressions"]:
            print("⚠️  PERFORMANCE REGRESSIONS DETECTED:")
            for regression in comparison["regressions"]:
                print(f"  - {regression['benchmark']}: "
                      f"+{regression['execution_time_change_percent']:.1f}% time, "
                      f"+{regression['memory_change_percent']:.1f}% memory")
            return 1
        else:
            print("✅ No significant performance regressions detected")
            
        if comparison["improvements"]:
            print("\n🚀 Performance Improvements:")
            for improvement in comparison["improvements"]:
                print(f"  - {improvement['benchmark']}: "
                      f"{improvement['execution_time_change_percent']:.1f}% time, "
                      f"{improvement['memory_change_percent']:.1f}% memory")
                      
        return 0
        
    if args.run_all:
        results = runner.run_full_benchmark_suite(args.iterations)
        results_file = runner.save_results(results, args.output)
        
        # Generate human-readable report
        report = runner.generate_benchmark_report(results)
        report_file = results_file.with_suffix('.md')
        with open(report_file, 'w') as f:
            f.write(report)
            
        print(f"Benchmark completed. Results saved to:")
        print(f"  JSON: {results_file}")
        print(f"  Report: {report_file}")
        
        return 0
        
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
