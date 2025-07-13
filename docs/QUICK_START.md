# LogGuard ML Quick Start Guide

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/SaahilParmar/LogGuard_ML.git
cd LogGuard_ML

# Install in development mode
pip install -e .

# Verify installation
logguard --version
```

## 📊 Basic Usage

### 1. Simple Log Analysis
```bash
# Basic analysis without ML
logguard analyze data/sample_log.log

# With ML anomaly detection
logguard analyze data/sample_log.log --ml
```

### 2. Advanced Analysis with Optimizations
```bash
# High-performance analysis with ensemble ML
logguard analyze large_log.log --ml \
  --algorithm ensemble \
  --parallel \
  --chunk-size 50000 \
  --profile \
  --verbose

# Custom output formats
logguard analyze app.log --ml --format json --output results.json
logguard analyze app.log --ml --format csv --output results.csv
```

### 3. Real-time Monitoring
```bash
# Monitor logs in real-time
logguard monitor /var/log/app.log --alerts

# With custom configuration
logguard monitor app.log --config production_config.yaml --alerts --verbose
```

### 4. Performance Profiling
```bash
# Profile all operations
logguard profile large_file.log --operations parse ml report

# Profile specific operations
logguard profile app.log --operations ml
```

## ⚙️ Configuration

Create a custom `config.yaml`:

```yaml
# Performance optimization
performance:
  use_parallel_parsing: true
  chunk_size: 50000
  max_workers: 8
  use_memory_optimization: true

# ML configuration
ml_model:
  algorithm: "ensemble"
  contamination: 0.1
  use_pca: true
  n_components: 50

# Real-time monitoring
monitoring:
  enabled: true
  buffer_size: 1000

# Alerting
alerting:
  enabled: true
  anomaly_threshold: 5
  email:
    enabled: true
    smtp_server: "smtp.gmail.com"
    from: "alerts@company.com"
    to: ["admin@company.com"]
```

## 🎯 Use Cases

### Production Log Monitoring
```bash
# Set up continuous monitoring with alerting
logguard monitor /var/log/production.log \
  --config production_config.yaml \
  --alerts \
  --buffer-size 500
```

### Batch Analysis of Historical Logs
```bash
# Analyze large historical log files
logguard analyze logs/*.log \
  --ml \
  --algorithm ensemble \
  --parallel \
  --chunk-size 100000 \
  --output historical_analysis.html
```

### Performance Testing
```bash
# Benchmark your system performance
python scripts/benchmark.py

# Profile specific files
logguard profile test_data.log --operations parse ml report
```

## 📈 Performance Tips

1. **For Large Files**: Use `--parallel` and adjust `--chunk-size`
2. **For Memory-Constrained Systems**: Use smaller chunk sizes and enable memory optimization
3. **For Real-time Processing**: Tune buffer sizes and use appropriate thresholds
4. **For Production**: Enable caching and use ensemble algorithms for best accuracy

## 🔧 Troubleshooting

### Common Issues
- **Import Errors**: Ensure all dependencies are installed with `pip install -e .`
- **Memory Issues**: Reduce chunk size or enable memory optimization
- **Performance Issues**: Use profiling to identify bottlenecks

### Getting Help
- Check the logs with `--verbose` flag
- Use `logguard --help` for command options
- Review configuration files for proper settings
