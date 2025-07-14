"""
Web-based Configuration UI for LogGuard ML

This module provides a simple Flask-based web interface for configuring
LogGuard ML settings, managing plugins, and monitoring system status.

Features:
- Configuration file editing with validation
- Plugin management interface
- System status monitoring
- Performance metrics dashboard
- Real-time log monitoring interface

Usage:
    python config_ui/app.py
    # Visit http://localhost:5000 in your browser
"""

import os
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from flask import (
        Flask, render_template, request, jsonify, redirect,
        url_for, flash, send_from_directory,
    )
    FLASK_AVAILABLE = True
except ImportError:
    print("Flask not available. Install with: pip install flask")
    FLASK_AVAILABLE = False
    Flask = None

from logguard_ml import __version__
from logguard_ml.plugins import plugin_manager

app = Flask(__name__) if FLASK_AVAILABLE else None
if app:
    app.secret_key = 'logguard-ml-config-ui-secret-key'

# Configuration file paths
CONFIG_DIR = Path(__file__).parent.parent / "config"
PLUGIN_DIR = Path(__file__).parent.parent / "plugins"
REPORTS_DIR = Path(__file__).parent.parent / "reports"


class ConfigManager:
    """Manager for configuration file operations."""
    
    def __init__(self):
        self.config_file = CONFIG_DIR / "config.yaml"
        self.backup_dir = CONFIG_DIR / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        
    def load_config(self) -> Dict:
        """Load current configuration."""
        try:
            with open(self.config_file) as f:
                return yaml.safe_load(f)
        except Exception as e:
            return {"error": f"Failed to load config: {e}"}
    
    def save_config(self, config: Dict) -> bool:
        """Save configuration with backup."""
        try:
            # Create backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_dir / f"config_backup_{timestamp}.yaml"
            
            if self.config_file.exists():
                import shutil
                shutil.copy2(self.config_file, backup_file)
            
            # Save new config
            with open(self.config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            return True
        except Exception as e:
            print(f"Failed to save config: {e}")
            return False
    
    def validate_config(self, config: Dict) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Check required sections
        required_sections = ['log_patterns', 'ml_model']
        for section in required_sections:
            if section not in config:
                issues.append(f"Missing required section: {section}")
        
        # Validate ML model settings
        if 'ml_model' in config:
            ml_config = config['ml_model']
            
            if 'contamination' in ml_config:
                contamination = ml_config['contamination']
                if not 0.01 <= contamination <= 0.5:
                    issues.append("Contamination must be between 0.01 and 0.5")
            
            if 'algorithm' in ml_config:
                valid_algorithms = ['isolation_forest', 'one_class_svm', 'local_outlier_factor', 'ensemble']
                if ml_config['algorithm'] not in valid_algorithms:
                    issues.append(f"Invalid algorithm. Must be one of: {valid_algorithms}")
        
        return issues


config_manager = ConfigManager()


# Route definitions
def index():
    """Main dashboard page."""
    return render_template('index.html', version=__version__)


def config_page():
    """Configuration editing page."""
    config = config_manager.load_config()
    return render_template('config.html', config=config)


def api_config():
    """API endpoint for configuration management."""
    if request.method == 'GET':
        return jsonify(config_manager.load_config())
    
    elif request.method == 'POST':
        try:
            new_config = request.json
            
            # Validate configuration
            issues = config_manager.validate_config(new_config)
            if issues:
                return jsonify({"success": False, "errors": issues}), 400
            
            # Save configuration
            if config_manager.save_config(new_config):
                return jsonify({"success": True, "message": "Configuration saved successfully"})
            else:
                return jsonify({"success": False, "error": "Failed to save configuration"}), 500
                
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500


def plugins_page():
    """Plugin management page."""
    # Load plugins
    plugin_manager.load_plugins_from_directory(PLUGIN_DIR)
    plugins = plugin_manager.list_plugins()
    
    # Get plugin metadata
    plugin_info = {}
    for category, plugin_list in plugins.items():
        plugin_info[category] = []
        for plugin_name in plugin_list:
            try:
                info = plugin_manager.get_plugin_info(plugin_name)
                plugin_info[category].append(info)
            except:
                plugin_info[category].append({"name": plugin_name, "error": "No metadata available"})
    
    return render_template('plugins.html', plugins=plugin_info)


def api_plugins():
    """API endpoint for plugin information."""
    plugin_manager.load_plugins_from_directory(PLUGIN_DIR)
    plugins = plugin_manager.list_plugins()
    
    return jsonify({
        "plugins": plugins,
        "total_count": sum(len(v) for v in plugins.values())
    })


def monitoring_page():
    """Real-time monitoring page."""
    return render_template('monitoring.html')


def api_system_status():
    """API endpoint for system status."""
    try:
        import psutil
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get LogGuard ML status
        config = config_manager.load_config()
        
        status = {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_percent": disk.percent,
                "disk_free_gb": round(disk.free / (1024**3), 2)
            },
            "logguard": {
                "version": __version__,
                "config_valid": len(config_manager.validate_config(config)) == 0,
                "plugins_loaded": sum(len(v) for v in plugin_manager.list_plugins().values())
            }
        }
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def reports_page():
    """Reports management page."""
    reports = []
    
    if REPORTS_DIR.exists():
        for report_file in REPORTS_DIR.glob("*.html"):
            try:
                stat = report_file.stat()
                reports.append({
                    "name": report_file.name,
                    "size_mb": round(stat.st_size / (1024*1024), 2),
                    "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                    "path": str(report_file.relative_to(Path.cwd()))
                })
            except:
                pass
    
    # Sort by modification time, newest first
    reports.sort(key=lambda x: x["modified"], reverse=True)
    
    return render_template('reports.html', reports=reports)


def serve_report(filename):
    """Serve report files."""
    return send_from_directory(REPORTS_DIR, filename)


def api_performance_metrics():
    """API endpoint for performance metrics."""
    # This would typically come from a metrics database
    # For demo purposes, generate sample data
    
    now = datetime.now()
    metrics = {
        "parsing_performance": [
            {
                "timestamp": (now - timedelta(minutes=i*5)).isoformat(),
                "logs_per_second": 1000 + (i * 50) + (i % 3) * 100,
                "memory_usage_mb": 150 + (i * 10) + (i % 2) * 20
            }
            for i in range(12)  # Last hour in 5-minute intervals
        ],
        "anomaly_detection": [
            {
                "timestamp": (now - timedelta(minutes=i*5)).isoformat(),
                "accuracy": 0.95 + (i % 3) * 0.01,
                "processing_time_ms": 200 + (i * 10) + (i % 4) * 50
            }
            for i in range(12)
        ]
    }
    
    return jsonify(metrics)


# Register routes if Flask is available
if app:
    app.route('/')(index)
    app.route('/config')(config_page)
    app.route('/api/config', methods=['GET', 'POST'])(api_config)
    app.route('/plugins')(plugins_page)
    app.route('/api/plugins')(api_plugins)
    app.route('/monitoring')(monitoring_page)
    app.route('/api/system_status')(api_system_status)
    app.route('/reports')(reports_page)
    app.route('/reports/<filename>')(serve_report)
    app.route('/api/performance_metrics')(api_performance_metrics)


def create_templates():
    """Create HTML templates for the web interface."""
    templates_dir = Path(__file__).parent / "templates"
    templates_dir.mkdir(exist_ok=True)
    
    # Base template
    base_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}LogGuard ML{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .sidebar { min-height: 100vh; background-color: #f8f9fa; }
        .nav-pills .nav-link.active { background-color: #0d6efd; }
        .card { border: none; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status-indicator { width: 12px; height: 12px; border-radius: 50%; display: inline-block; }
        .status-success { background-color: #28a745; }
        .status-warning { background-color: #ffc107; }
        .status-danger { background-color: #dc3545; }
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="row">
            <!-- Sidebar -->
            <nav class="col-md-2 sidebar p-3">
                <h5><i class="fas fa-shield-alt"></i> LogGuard ML</h5>
                <ul class="nav nav-pills flex-column">
                    <li class="nav-item">
                        <a class="nav-link {% if request.endpoint == 'index' %}active{% endif %}" href="{{ url_for('index') }}">
                            <i class="fas fa-home"></i> Dashboard
                        </a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link {% if request.endpoint == 'config_page' %}active{% endif %}" href="{{ url_for('config_page') }}">
                            <i class="fas fa-cog"></i> Configuration
                        </a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link {% if request.endpoint == 'plugins_page' %}active{% endif %}" href="{{ url_for('plugins_page') }}">
                            <i class="fas fa-puzzle-piece"></i> Plugins
                        </a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link {% if request.endpoint == 'monitoring_page' %}active{% endif %}" href="{{ url_for('monitoring_page') }}">
                            <i class="fas fa-chart-line"></i> Monitoring
                        </a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link {% if request.endpoint == 'reports_page' %}active{% endif %}" href="{{ url_for('reports_page') }}">
                            <i class="fas fa-file-alt"></i> Reports
                        </a>
                    </li>
                </ul>
            </nav>
            
            <!-- Main content -->
            <main class="col-md-10 p-4">
                {% with messages = get_flashed_messages() %}
                    {% if messages %}
                        {% for message in messages %}
                            <div class="alert alert-info alert-dismissible fade show" role="alert">
                                {{ message }}
                                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                            </div>
                        {% endfor %}
                    {% endif %}
                {% endwith %}
                
                {% block content %}{% endblock %}
            </main>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    {% block scripts %}{% endblock %}
</body>
</html>"""
    
    # Index template
    index_template = """{% extends "base.html" %}

{% block title %}Dashboard - LogGuard ML{% endblock %}

{% block content %}
<div class="d-flex justify-content-between align-items-center mb-4">
    <h1><i class="fas fa-home"></i> Dashboard</h1>
    <span class="badge bg-secondary">v{{ version }}</span>
</div>

<div class="row">
    <div class="col-md-3">
        <div class="card text-center">
            <div class="card-body">
                <i class="fas fa-shield-alt fa-2x text-primary mb-2"></i>
                <h5>System Status</h5>
                <span class="status-indicator status-success"></span>
                <span class="ms-2">Operational</span>
            </div>
        </div>
    </div>
    <div class="col-md-3">
        <div class="card text-center">
            <div class="card-body">
                <i class="fas fa-puzzle-piece fa-2x text-info mb-2"></i>
                <h5>Plugins</h5>
                <h3 id="plugin-count">-</h3>
                <small>Loaded</small>
            </div>
        </div>
    </div>
    <div class="col-md-3">
        <div class="card text-center">
            <div class="card-body">
                <i class="fas fa-cpu fa-2x text-warning mb-2"></i>
                <h5>CPU Usage</h5>
                <h3 id="cpu-usage">-</h3>
                <small>Percent</small>
            </div>
        </div>
    </div>
    <div class="col-md-3">
        <div class="card text-center">
            <div class="card-body">
                <i class="fas fa-memory fa-2x text-danger mb-2"></i>
                <h5>Memory</h5>
                <h3 id="memory-usage">-</h3>
                <small>Percent</small>
            </div>
        </div>
    </div>
</div>

<div class="row mt-4">
    <div class="col-md-12">
        <div class="card">
            <div class="card-header">
                <h5><i class="fas fa-chart-line"></i> System Metrics</h5>
            </div>
            <div class="card-body">
                <canvas id="metricsChart" height="100"></canvas>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
// Update system status
function updateSystemStatus() {
    fetch('/api/system_status')
        .then(response => response.json())
        .then(data => {
            document.getElementById('cpu-usage').textContent = data.system.cpu_percent + '%';
            document.getElementById('memory-usage').textContent = data.system.memory_percent + '%';
        });
    
    fetch('/api/plugins')
        .then(response => response.json())
        .then(data => {
            document.getElementById('plugin-count').textContent = data.total_count;
        });
}

// Initialize chart
const ctx = document.getElementById('metricsChart').getContext('2d');
const chart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'CPU Usage %',
            data: [],
            borderColor: 'rgb(255, 193, 7)',
            tension: 0.1
        }, {
            label: 'Memory Usage %',
            data: [],
            borderColor: 'rgb(220, 53, 69)',
            tension: 0.1
        }]
    },
    options: {
        responsive: true,
        scales: {
            y: {
                beginAtZero: true,
                max: 100
            }
        }
    }
});

// Update every 5 seconds
setInterval(updateSystemStatus, 5000);
updateSystemStatus();
</script>
{% endblock %}"""
    
    # Save templates
    with open(templates_dir / "base.html", "w") as f:
        f.write(base_template)
    
    with open(templates_dir / "index.html", "w") as f:
        f.write(index_template)


def main():
    """Main entry point for the configuration UI."""
    if not FLASK_AVAILABLE:
        print("Error: Flask is required for the configuration UI")
        print("Install with: pip install flask")
        return 1
    
    # Create templates
    create_templates()
    
    print("LogGuard ML Configuration UI")
    print("=" * 40)
    print(f"Version: {__version__}")
    print("Starting web server on http://localhost:5000")
    print("Press Ctrl+C to stop")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\nShutting down...")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
