# LogGuard ML Production Deployment Guide

## 🏗️ Production Setup

### System Requirements
- **CPU**: 4+ cores recommended for parallel processing
- **Memory**: 8GB+ RAM for large log files
- **Storage**: SSD recommended for better I/O performance
- **Python**: 3.9+ with virtual environment

### 1. Environment Setup

```bash
# Create production environment
python -m venv /opt/logguard_ml
source /opt/logguard_ml/bin/activate

# Install LogGuard ML
pip install -e .

# Verify installation
logguard --version
```

### 2. Configuration Management

Create production configuration at `/etc/logguard_ml/config.yaml`:

```yaml
# Production configuration
performance:
  use_parallel_parsing: true
  chunk_size: 100000
  max_workers: 8
  use_memory_optimization: true

ml_model:
  algorithm: "ensemble"
  contamination: 0.05  # Lower threshold for production
  use_pca: true
  n_components: 100

monitoring:
  enabled: true
  buffer_size: 1000

alerting:
  enabled: true
  anomaly_threshold: 3
  time_window_minutes: 5
  throttle_minutes: 15
  
  email:
    enabled: true
    smtp_server: "smtp.company.com"
    smtp_port: 587
    username: "logguard@company.com"
    password: "${SMTP_PASSWORD}"
    from: "logguard@company.com"
    to: 
      - "ops-team@company.com"
      - "security@company.com"
  
  webhook:
    enabled: true
    url: "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
    timeout: 30

logging:
  level: "INFO"
  file: "/var/log/logguard_ml.log"
  max_size: "100MB"
  backup_count: 5
```

### 3. Systemd Service Setup

Create `/etc/systemd/system/logguard-monitor.service`:

```ini
[Unit]
Description=LogGuard ML Real-time Log Monitor
After=network.target

[Service]
Type=simple
User=logguard
Group=logguard
WorkingDirectory=/opt/logguard_ml
Environment=PATH=/opt/logguard_ml/bin
ExecStart=/opt/logguard_ml/bin/logguard monitor /var/log/application.log --config /etc/logguard_ml/config.yaml --alerts
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable logguard-monitor
sudo systemctl start logguard-monitor
```

### 4. Log Rotation Setup

Create `/etc/logrotate.d/logguard_ml`:

```
/var/log/logguard_ml.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    postrotate
        systemctl reload logguard-monitor
    endscript
}
```

### 5. Monitoring & Health Checks

```bash
# Service status
sudo systemctl status logguard-monitor

# View logs
sudo journalctl -u logguard-monitor -f

# Performance monitoring
logguard profile /var/log/application.log --operations parse ml
```

## 🔄 Automated Deployment

### Using Docker

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install -e .

EXPOSE 8000

CMD ["logguard", "monitor", "/logs/app.log", "--config", "/config/config.yaml", "--alerts"]
```

### Using Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'
services:
  logguard:
    build: .
    volumes:
      - /var/log:/logs:ro
      - ./config:/config:ro
      - logguard_data:/data
    environment:
      - SMTP_PASSWORD=${SMTP_PASSWORD}
    restart: unless-stopped

volumes:
  logguard_data:
```

## 📊 Performance Optimization

### 1. Memory Tuning
```yaml
performance:
  chunk_size: 50000  # Adjust based on available memory
  max_workers: 4     # Number of CPU cores
  use_memory_optimization: true
```

### 2. ML Model Selection
```yaml
ml_model:
  # For high throughput
  algorithm: "isolation_forest"
  
  # For best accuracy
  algorithm: "ensemble"
  
  # For balanced performance
  algorithm: "one_class_svm"
```

### 3. Caching Configuration
```yaml
caching:
  enabled: true
  max_size: "1GB"
  ttl_hours: 24
```

## 🚨 Alerting Configuration

### Email Alerts
```yaml
alerting:
  email:
    enabled: true
    templates:
      subject: "🚨 LogGuard Alert: {{anomaly_count}} anomalies detected"
      body: |
        Anomalies detected in {{log_file}}:
        - Count: {{anomaly_count}}
        - Time window: {{time_window}}
        - Severity: {{severity}}
        
        View details: {{report_url}}
```

### Slack Integration
```yaml
alerting:
  webhook:
    enabled: true
    url: "https://hooks.slack.com/services/..."
    payload:
      channel: "#ops-alerts"
      username: "LogGuard ML"
      icon_emoji: ":warning:"
      text: "🚨 {{anomaly_count}} anomalies detected in {{log_file}}"
```

## 🔧 Maintenance

### Regular Tasks
1. **Log Rotation**: Ensure logs don't fill disk space
2. **Model Updates**: Retrain models with new data periodically
3. **Performance Review**: Monitor system resources and adjust settings
4. **Security Updates**: Keep dependencies updated

### Backup Strategy
```bash
# Backup configuration
tar -czf logguard_backup_$(date +%Y%m%d).tar.gz /etc/logguard_ml/

# Backup model cache
tar -czf model_cache_$(date +%Y%m%d).tar.gz /var/cache/logguard_ml/
```

## 📈 Scaling

### Horizontal Scaling
- Deploy multiple instances for different log sources
- Use load balancer for API endpoints
- Centralize configuration management

### Vertical Scaling
- Increase CPU cores for parallel processing
- Add memory for larger datasets
- Use faster storage (NVMe SSD)

## 🛡️ Security

### Access Control
```bash
# Create dedicated user
sudo useradd -r -s /bin/false logguard
sudo chown -R logguard:logguard /opt/logguard_ml
```

### Network Security
- Restrict network access to monitoring ports
- Use TLS for webhook communications
- Secure SMTP credentials with environment variables

## 📊 Monitoring LogGuard ML

### Metrics to Monitor
- Processing throughput (entries/second)
- Memory usage patterns
- Alert frequency and accuracy
- System resource utilization

### Health Check Endpoint
Add to your monitoring system:
```bash
# Check service status
systemctl is-active logguard-monitor

# Check processing health
tail -n 100 /var/log/logguard_ml.log | grep -c "ERROR"
```
