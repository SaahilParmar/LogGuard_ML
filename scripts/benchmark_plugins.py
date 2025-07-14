#!/usr/bin/env python3
"""
Plugin Performance Benchmarking Script

This script provides comprehensive performance benchmarking for LogGuard ML plugins,
helping developers evaluate and optimize their plugin implementations.

Features:
- Memory usage profiling
- Execution time measurement
- Scalability testing
- Comparative analysis
- Resource utilization monitoring
"""

import time
import psutil
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import json
import tracemalloc
from datetime import datetime
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from logguard_ml.plugins import PluginManager, MLDetectorPlugin, OutputFormatPlugin


class PluginBenchmark:
    """
    Comprehensive benchmarking suite for LogGuard ML plugins.
    """
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        self.test_datasets = {}
        
        # Initialize plugin manager
        self.plugin_manager = PluginManager()
        
        # Setup plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def generate_test_datasets(self) -> None:
        """Generate test datasets of various sizes."""
        print("Generating test datasets...")
        
        # Small dataset (1K records)
        self.test_datasets['small'] = self._create_synthetic_logs(1000)
        
        # Medium dataset (10K records)
        self.test_datasets['medium'] = self._create_synthetic_logs(10000)
        
        # Large dataset (100K records)
        self.test_datasets['large'] = self._create_synthetic_logs(100000)
        
        # XL dataset (1M records) - only if memory allows
        try:
            self.test_datasets['xl'] = self._create_synthetic_logs(1000000)
        except MemoryError:
            print("Warning: Skipping XL dataset due to memory constraints")
    
    def _create_synthetic_logs(self, n_records: int) -> pd.DataFrame:
        """Create synthetic log data for testing."""
        np.random.seed(42)  # For reproducible results
        
        # Generate realistic log patterns
        levels = np.random.choice(['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                                 size=n_records, p=[0.3, 0.5, 0.15, 0.05])
        
        # Generate timestamps (last 30 days)
        start_time = pd.Timestamp.now() - pd.Timedelta(days=30)
        timestamps = pd.date_range(start=start_time, periods=n_records, freq='S')
        
        # Generate messages with varying complexity
        base_messages = [
            "User login successful",
            "Database connection established", 
            "Request processed successfully",
            "Cache miss for key",
            "Authentication failed for user",
            "Database query took {time}ms",
            "Memory usage at {percent}%",
            "Network timeout occurred",
            "Invalid request format received",
            "System backup completed"
        ]
        
        messages = []
        for _ in range(n_records):
            base_msg = np.random.choice(base_messages)
            # Add some randomization
            if '{time}' in base_msg:
                base_msg = base_msg.format(time=np.random.randint(10, 5000))
            elif '{percent}' in base_msg:
                base_msg = base_msg.format(percent=np.random.randint(10, 95))
            
            # Occasionally add longer, more complex messages
            if np.random.random() < 0.1:
                base_msg += f" Additional context: {np.random.choice(['session_id=12345', 'ip=192.168.1.100', 'user_id=user123'])}"
            
            messages.append(base_msg)
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'level': levels,
            'message': messages
        })
    
    def benchmark_ml_detector(self, detector_name: str, plugin_class: type) -> Dict[str, Any]:
        """Benchmark ML detector plugin performance."""
        print(f"\nBenchmarking ML Detector: {detector_name}")
        
        # Register plugin
        self.plugin_manager.register_ml_detector(plugin_class)
        detector = self.plugin_manager.get_ml_detector(detector_name)
        
        results = {
            'plugin_name': detector_name,
            'plugin_type': 'ml_detector',
            'datasets': {},
            'memory_usage': {},
            'scalability': {}
        }
        
        for dataset_name, dataset in self.test_datasets.items():
            print(f"  Testing on {dataset_name} dataset ({len(dataset)} records)...")
            
            # Memory tracking
            tracemalloc.start()
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Performance measurement
            start_time = time.time()
            
            try:
                # Run detection
                result_df = detector.detect_anomalies(dataset, {})
                
                # Measure time and memory
                end_time = time.time()
                execution_time = end_time - start_time
                
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_delta = final_memory - initial_memory
                peak_memory = peak / 1024 / 1024  # MB
                
                # Validate results
                anomaly_count = result_df['anomaly'].sum() if 'anomaly' in result_df.columns else 0
                
                results['datasets'][dataset_name] = {
                    'execution_time': execution_time,
                    'memory_delta': memory_delta,
                    'peak_memory': peak_memory,
                    'records_processed': len(dataset),
                    'anomalies_detected': int(anomaly_count),
                    'throughput': len(dataset) / execution_time,  # records/second
                    'success': True
                }
                
                print(f"    ✓ Processed {len(dataset)} records in {execution_time:.2f}s")
                print(f"    ✓ Memory delta: {memory_delta:.1f}MB, Peak: {peak_memory:.1f}MB")
                print(f"    ✓ Throughput: {len(dataset)/execution_time:.0f} records/sec")
                
            except Exception as e:
                print(f"    ✗ Error: {str(e)}")
                results['datasets'][dataset_name] = {
                    'error': str(e),
                    'success': False
                }
            
            # Clean up memory
            gc.collect()
        
        # Calculate scalability metrics
        successful_tests = {k: v for k, v in results['datasets'].items() if v.get('success', False)}
        if len(successful_tests) >= 2:
            # Calculate how execution time scales with data size
            sizes = [successful_tests[k]['records_processed'] for k in successful_tests]
            times = [successful_tests[k]['execution_time'] for k in successful_tests]
            
            # Simple linear regression to estimate scalability
            if len(sizes) > 1:
                coeffs = np.polyfit(sizes, times, 1)
                results['scalability'] = {
                    'linear_coefficient': float(coeffs[0]),
                    'time_complexity': self._estimate_time_complexity(sizes, times),
                    'efficiency_rating': self._calculate_efficiency_rating(sizes, times)
                }
        
        return results
    
    def benchmark_output_format(self, format_name: str, plugin_class: type) -> Dict[str, Any]:
        """Benchmark output format plugin performance."""
        print(f"\nBenchmarking Output Format: {format_name}")
        
        # Register plugin
        self.plugin_manager.register_output_format(plugin_class)
        formatter = self.plugin_manager.get_output_format(format_name)
        
        results = {
            'plugin_name': format_name,
            'plugin_type': 'output_format',
            'datasets': {}
        }
        
        for dataset_name, dataset in self.test_datasets.items():
            print(f"  Testing on {dataset_name} dataset ({len(dataset)} records)...")
            
            # Add dummy anomaly data for realistic testing
            test_data = dataset.copy()
            test_data['anomaly'] = np.random.choice([True, False], size=len(dataset), p=[0.1, 0.9])
            test_data['anomaly_score'] = np.random.uniform(0, 1, size=len(dataset))
            
            # Create temporary output file
            output_file = self.output_dir / f"test_output_{format_name}_{dataset_name}.{formatter.file_extension}"
            
            try:
                start_time = time.time()
                
                # Generate output
                formatter.generate_output(test_data, str(output_file))
                
                end_time = time.time()
                execution_time = end_time - start_time
                
                # Get file size
                file_size = output_file.stat().st_size / 1024  # KB
                
                results['datasets'][dataset_name] = {
                    'execution_time': execution_time,
                    'output_file_size_kb': file_size,
                    'records_processed': len(dataset),
                    'throughput': len(dataset) / execution_time,
                    'compression_ratio': len(dataset) / file_size if file_size > 0 else 0,
                    'success': True
                }
                
                print(f"    ✓ Generated output in {execution_time:.2f}s")
                print(f"    ✓ Output size: {file_size:.1f}KB")
                print(f"    ✓ Throughput: {len(dataset)/execution_time:.0f} records/sec")
                
                # Clean up test file
                output_file.unlink()
                
            except Exception as e:
                print(f"    ✗ Error: {str(e)}")
                results['datasets'][dataset_name] = {
                    'error': str(e),
                    'success': False
                }
        
        return results
    
    def _estimate_time_complexity(self, sizes: List[int], times: List[float]) -> str:
        """Estimate the time complexity of the algorithm."""
        if len(sizes) < 2:
            return "insufficient_data"
        
        # Sort by size
        sorted_data = sorted(zip(sizes, times))
        sizes_sorted = [x[0] for x in sorted_data]
        times_sorted = [x[1] for x in sorted_data]
        
        # Calculate growth ratios
        size_ratios = [sizes_sorted[i] / sizes_sorted[i-1] for i in range(1, len(sizes_sorted))]
        time_ratios = [times_sorted[i] / times_sorted[i-1] for i in range(1, len(times_sorted))]
        
        if not size_ratios or not time_ratios:
            return "insufficient_data"
        
        avg_size_ratio = np.mean(size_ratios)
        avg_time_ratio = np.mean(time_ratios)
        
        # Estimate complexity based on ratios
        if avg_time_ratio <= avg_size_ratio * 1.2:
            return "O(n) - Linear"
        elif avg_time_ratio <= avg_size_ratio ** 1.5 * 1.2:
            return "O(n log n) - Linearithmic"
        elif avg_time_ratio <= avg_size_ratio ** 2 * 1.2:
            return "O(n²) - Quadratic"
        else:
            return "O(n³+) - Polynomial or worse"
    
    def _calculate_efficiency_rating(self, sizes: List[int], times: List[float]) -> str:
        """Calculate efficiency rating based on performance."""
        if not sizes or not times:
            return "unknown"
        
        # Calculate average throughput
        throughputs = [s / t for s, t in zip(sizes, times)]
        avg_throughput = np.mean(throughputs)
        
        # Rate based on throughput (records per second)
        if avg_throughput > 100000:
            return "excellent"
        elif avg_throughput > 50000:
            return "very_good"
        elif avg_throughput > 10000:
            return "good"
        elif avg_throughput > 1000:
            return "fair"
        else:
            return "poor"
    
    def run_comprehensive_benchmark(self, plugins: Dict[str, Dict[str, type]]) -> None:
        """Run comprehensive benchmark on all provided plugins."""
        print("Starting comprehensive plugin benchmark...")
        
        # Generate test datasets
        self.generate_test_datasets()
        
        benchmark_results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'system_info': self._get_system_info(),
                'datasets': {name: len(data) for name, data in self.test_datasets.items()}
            },
            'results': {}
        }
        
        # Benchmark ML detectors
        if 'ml_detectors' in plugins:
            print("\n" + "="*50)
            print("BENCHMARKING ML DETECTORS")
            print("="*50)
            
            for detector_name, detector_class in plugins['ml_detectors'].items():
                try:
                    result = self.benchmark_ml_detector(detector_name, detector_class)
                    benchmark_results['results'][detector_name] = result
                except Exception as e:
                    print(f"Error benchmarking {detector_name}: {e}")
                    benchmark_results['results'][detector_name] = {'error': str(e)}
        
        # Benchmark output formats
        if 'output_formats' in plugins:
            print("\n" + "="*50)
            print("BENCHMARKING OUTPUT FORMATS")
            print("="*50)
            
            for format_name, format_class in plugins['output_formats'].items():
                try:
                    result = self.benchmark_output_format(format_name, format_class)
                    benchmark_results['results'][format_name] = result
                except Exception as e:
                    print(f"Error benchmarking {format_name}: {e}")
                    benchmark_results['results'][format_name] = {'error': str(e)}
        
        # Save results
        results_file = self.output_dir / f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2, default=str)
        
        print(f"\nBenchmark results saved to: {results_file}")
        
        # Generate visualizations
        self.generate_visualizations(benchmark_results)
        
        # Generate summary report
        self.generate_summary_report(benchmark_results)
        
        return benchmark_results
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for context."""
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': sys.version,
            'platform': sys.platform
        }
    
    def generate_visualizations(self, results: Dict[str, Any]) -> None:
        """Generate performance visualization charts."""
        print("\nGenerating visualizations...")
        
        # Extract performance data
        performance_data = []
        
        for plugin_name, plugin_results in results['results'].items():
            if 'datasets' not in plugin_results:
                continue
            
            for dataset_name, dataset_results in plugin_results['datasets'].items():
                if dataset_results.get('success', False):
                    performance_data.append({
                        'plugin': plugin_name,
                        'dataset': dataset_name,
                        'records': dataset_results['records_processed'],
                        'time': dataset_results['execution_time'],
                        'throughput': dataset_results['throughput'],
                        'type': plugin_results['plugin_type']
                    })
        
        if not performance_data:
            print("No data available for visualization")
            return
        
        df = pd.DataFrame(performance_data)
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('LogGuard ML Plugin Performance Benchmark', fontsize=16, fontweight='bold')
        
        # 1. Execution Time vs Dataset Size
        for plugin in df['plugin'].unique():
            plugin_data = df[df['plugin'] == plugin]
            ax1.plot(plugin_data['records'], plugin_data['time'], 
                    marker='o', linewidth=2, label=plugin)
        ax1.set_xlabel('Number of Records')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('Execution Time vs Dataset Size')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Throughput Comparison
        throughput_data = df.groupby('plugin')['throughput'].mean().sort_values(ascending=True)
        bars = ax2.barh(range(len(throughput_data)), throughput_data.values)
        ax2.set_yticks(range(len(throughput_data)))
        ax2.set_yticklabels(throughput_data.index)
        ax2.set_xlabel('Average Throughput (records/sec)')
        ax2.set_title('Average Throughput by Plugin')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, throughput_data.values)):
            ax2.text(value + max(throughput_data.values) * 0.01, i, 
                    f'{value:.0f}', va='center', fontweight='bold')
        
        # 3. Scalability Analysis
        for plugin in df['plugin'].unique():
            plugin_data = df[df['plugin'] == plugin].sort_values('records')
            if len(plugin_data) > 1:
                # Calculate relative performance (normalized)
                baseline = plugin_data.iloc[0]['time'] / plugin_data.iloc[0]['records']
                relative_perf = (plugin_data['time'] / plugin_data['records']) / baseline
                ax3.plot(plugin_data['records'], relative_perf, 
                        marker='s', linewidth=2, label=plugin)
        
        ax3.set_xlabel('Number of Records')
        ax3.set_ylabel('Relative Performance (normalized)')
        ax3.set_title('Scalability Analysis')
        ax3.set_xscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Baseline')
        
        # 4. Plugin Type Comparison
        type_performance = df.groupby(['type', 'plugin'])['throughput'].mean().reset_index()
        sns.boxplot(data=type_performance, x='type', y='throughput', ax=ax4)
        ax4.set_title('Throughput Distribution by Plugin Type')
        ax4.set_ylabel('Throughput (records/sec)')
        
        # Adjust layout and save
        plt.tight_layout()
        chart_file = self.output_dir / f"benchmark_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to: {chart_file}")
    
    def generate_summary_report(self, results: Dict[str, Any]) -> None:
        """Generate a summary report of benchmark results."""
        print("\nGenerating summary report...")
        
        report_lines = [
            "# LogGuard ML Plugin Benchmark Report",
            f"Generated on: {results['metadata']['timestamp']}",
            "",
            "## System Information",
            f"- CPU Cores: {results['metadata']['system_info']['cpu_count']}",
            f"- Total Memory: {results['metadata']['system_info']['memory_total_gb']:.1f} GB",
            f"- Python Version: {results['metadata']['system_info']['python_version']}",
            f"- Platform: {results['metadata']['system_info']['platform']}",
            "",
            "## Test Datasets",
        ]
        
        for dataset_name, size in results['metadata']['datasets'].items():
            report_lines.append(f"- {dataset_name.title()}: {size:,} records")
        
        report_lines.extend([
            "",
            "## Plugin Performance Summary",
            ""
        ])
        
        # Analyze results
        performance_summary = []
        
        for plugin_name, plugin_results in results['results'].items():
            if 'error' in plugin_results:
                report_lines.append(f"### {plugin_name} (ERROR)")
                report_lines.append(f"Error: {plugin_results['error']}")
                report_lines.append("")
                continue
            
            report_lines.append(f"### {plugin_name}")
            report_lines.append(f"Type: {plugin_results['plugin_type']}")
            report_lines.append("")
            
            # Calculate average metrics
            successful_tests = [v for v in plugin_results['datasets'].values() if v.get('success', False)]
            
            if successful_tests:
                avg_throughput = np.mean([t['throughput'] for t in successful_tests])
                total_records = sum([t['records_processed'] for t in successful_tests])
                total_time = sum([t['execution_time'] for t in successful_tests])
                
                report_lines.extend([
                    f"- **Average Throughput**: {avg_throughput:.0f} records/second",
                    f"- **Total Records Processed**: {total_records:,}",
                    f"- **Total Execution Time**: {total_time:.2f} seconds",
                ])
                
                if 'scalability' in plugin_results:
                    scalability = plugin_results['scalability']
                    report_lines.extend([
                        f"- **Time Complexity**: {scalability.get('time_complexity', 'Unknown')}",
                        f"- **Efficiency Rating**: {scalability.get('efficiency_rating', 'Unknown').title()}",
                    ])
                
                # Dataset-specific results
                report_lines.append("\n**Dataset Results:**")
                for dataset_name, dataset_result in plugin_results['datasets'].items():
                    if dataset_result.get('success', False):
                        report_lines.append(
                            f"- {dataset_name.title()}: {dataset_result['throughput']:.0f} rec/sec "
                            f"({dataset_result['execution_time']:.2f}s for {dataset_result['records_processed']:,} records)"
                        )
                
                performance_summary.append({
                    'plugin': plugin_name,
                    'type': plugin_results['plugin_type'],
                    'avg_throughput': avg_throughput,
                    'efficiency': scalability.get('efficiency_rating', 'unknown') if 'scalability' in plugin_results else 'unknown'
                })
            
            report_lines.append("")
        
        # Add recommendations
        if performance_summary:
            report_lines.extend([
                "## Recommendations",
                ""
            ])
            
            # Find best performers
            ml_detectors = [p for p in performance_summary if p['type'] == 'ml_detector']
            output_formats = [p for p in performance_summary if p['type'] == 'output_format']
            
            if ml_detectors:
                best_ml = max(ml_detectors, key=lambda x: x['avg_throughput'])
                report_lines.append(f"- **Best ML Detector**: {best_ml['plugin']} ({best_ml['avg_throughput']:.0f} rec/sec)")
            
            if output_formats:
                best_output = max(output_formats, key=lambda x: x['avg_throughput'])
                report_lines.append(f"- **Best Output Format**: {best_output['plugin']} ({best_output['avg_throughput']:.0f} rec/sec)")
            
            # General recommendations
            report_lines.extend([
                "",
                "### General Recommendations:",
                "- For high-volume environments, prioritize plugins with 'excellent' or 'very_good' efficiency ratings",
                "- Monitor memory usage with large datasets (>100K records)",
                "- Consider plugin combinations based on your specific use case",
                "- Regular benchmarking helps identify performance regressions",
            ])
        
        # Write report
        report_file = self.output_dir / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Summary report saved to: {report_file}")


def main():
    """Main benchmarking function with example plugins."""
    
    # Example ML Detector for testing
    class ExampleDetector(MLDetectorPlugin):
        @property
        def name(self) -> str:
            return "example_detector"
        
        @property
        def version(self) -> str:
            return "1.0.0"
        
        @property
        def description(self) -> str:
            return "Example detector for benchmarking"
        
        def detect_anomalies(self, df: pd.DataFrame, config: dict) -> pd.DataFrame:
            # Simple statistical outlier detection
            result = df.copy()
            result['anomaly'] = False
            result['anomaly_score'] = 0.5
            
            # Mark long messages as anomalies
            if 'message' in df.columns:
                long_messages = df['message'].str.len() > df['message'].str.len().quantile(0.95)
                result.loc[long_messages, 'anomaly'] = True
                result.loc[long_messages, 'anomaly_score'] = 0.8
            
            return result
    
    # Example Output Format for testing
    class ExampleFormat(OutputFormatPlugin):
        @property
        def name(self) -> str:
            return "example_format"
        
        @property
        def version(self) -> str:
            return "1.0.0"
        
        @property
        def description(self) -> str:
            return "Example format for benchmarking"
        
        @property
        def file_extension(self) -> str:
            return "txt"
        
        def generate_output(self, df: pd.DataFrame, output_path: str, **kwargs) -> None:
            anomaly_count = int(df['anomaly'].sum()) if 'anomaly' in df.columns else 0
            summary = f"Processed {len(df)} records, found {anomaly_count} anomalies"
            with open(output_path, 'w') as f:
                f.write(summary)
    
    # Set up benchmark
    benchmark = PluginBenchmark()
    
    # Define plugins to test
    plugins_to_test = {
        'ml_detectors': {
            'example_detector': ExampleDetector
        },
        'output_formats': {
            'example_format': ExampleFormat
        }
    }
    
    # Run benchmark
    print("LogGuard ML Plugin Benchmarking Tool")
    print("=" * 40)
    
    results = benchmark.run_comprehensive_benchmark(plugins_to_test)
    
    print("\nBenchmarking completed!")
    print(f"Results saved in: {benchmark.output_dir}")


if __name__ == "__main__":
    main()
