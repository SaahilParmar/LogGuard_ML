"""
Advanced Output Format Plugins for LogGuard ML

This module provides sophisticated output formats that demonstrate
the extensibility of the LogGuard ML reporting system.

Featured Formats:
- InteractiveDashboardFormat: Interactive HTML dashboard
- TimeSeriesFormat: Time-series visualization format
- ExecutiveSummaryFormat: Executive summary reports
- AlertFormat: Real-time alert formatting
- ComplianceReportFormat: Compliance and audit reports
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from logguard_ml.plugins import OutputFormatPlugin


class InteractiveDashboardFormat(OutputFormatPlugin):
    """
    Interactive HTML dashboard with advanced visualizations.
    
    Creates a comprehensive dashboard with multiple charts, filtering
    capabilities, and real-time updates.
    """
    
    @property
    def name(self) -> str:
        return "interactive_dashboard"
    
    @property
    def version(self) -> str:
        return "2.0.0"
    
    @property
    def description(self) -> str:
        return "Interactive HTML dashboard with advanced analytics"
    
    @property
    def file_extension(self) -> str:
        return "html"
    
    def _generate_dashboard_template(self) -> str:
        """Generate the base HTML template for the dashboard."""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LogGuard ML - Interactive Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .chart-container { min-height: 400px; margin: 20px 0; }
        .metric-card { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .metric-value { font-size: 2.5rem; font-weight: bold; }
        .metric-label { font-size: 0.9rem; opacity: 0.8; }
        .sidebar { 
            background: #f8f9fa; 
            min-height: 100vh; 
            padding: 20px;
            border-right: 1px solid #dee2e6;
        }
        .main-content { padding: 20px; }
        .alert-item {
            border-left: 4px solid #dc3545;
            padding: 10px;
            margin: 5px 0;
            background: #f8d7da;
            border-radius: 0 5px 5px 0;
        }
        .filter-section {
            background: white;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="row">
            <div class="col-md-3 sidebar">
                <h3>LogGuard ML</h3>
                <p class="text-muted">Interactive Analytics Dashboard</p>
                
                <div class="filter-section">
                    <h5>Filters</h5>
                    <div class="mb-3">
                        <label class="form-label">Time Range</label>
                        <select class="form-select" id="timeRange">
                            <option value="1h">Last Hour</option>
                            <option value="24h" selected>Last 24 Hours</option>
                            <option value="7d">Last 7 Days</option>
                            <option value="30d">Last 30 Days</option>
                        </select>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">Log Level</label>
                        <select class="form-select" id="logLevel">
                            <option value="all" selected>All Levels</option>
                            <option value="ERROR">ERROR</option>
                            <option value="WARNING">WARNING</option>
                            <option value="INFO">INFO</option>
                            <option value="DEBUG">DEBUG</option>
                        </select>
                    </div>
                    <button class="btn btn-primary" onclick="updateDashboard()">Apply Filters</button>
                </div>
                
                <div class="filter-section">
                    <h5>Quick Stats</h5>
                    <div class="metric-card">
                        <div class="metric-value" id="totalLogs">--</div>
                        <div class="metric-label">Total Logs</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="totalAnomalies">--</div>
                        <div class="metric-label">Anomalies Detected</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="anomalyRate">--%</div>
                        <div class="metric-label">Anomaly Rate</div>
                    </div>
                </div>
                
                <div class="filter-section">
                    <h5>Recent Alerts</h5>
                    <div id="recentAlerts">
                        <!-- Dynamic alerts will be inserted here -->
                    </div>
                </div>
            </div>
            
            <div class="col-md-9 main-content">
                <div class="d-flex justify-content-between align-items-center mb-4">
                    <h2>Log Analysis Dashboard</h2>
                    <div>
                        <button class="btn btn-outline-secondary" onclick="exportData()">Export Data</button>
                        <button class="btn btn-success" onclick="refreshData()">Refresh</button>
                    </div>
                </div>
                
                <div class="row">
                    <div class="col-12">
                        <div class="chart-container">
                            <h4>Anomaly Timeline</h4>
                            <div id="timelineChart"></div>
                        </div>
                    </div>
                </div>
                
                <div class="row">
                    <div class="col-md-6">
                        <div class="chart-container">
                            <h4>Log Level Distribution</h4>
                            <div id="levelChart"></div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="chart-container">
                            <h4>Anomaly Score Distribution</h4>
                            <div id="scoreChart"></div>
                        </div>
                    </div>
                </div>
                
                <div class="row">
                    <div class="col-12">
                        <div class="chart-container">
                            <h4>Top Anomalous Messages</h4>
                            <div id="anomaliesTable"></div>
                        </div>
                    </div>
                </div>
                
                <div class="row">
                    <div class="col-md-6">
                        <div class="chart-container">
                            <h4>Hourly Pattern Analysis</h4>
                            <div id="hourlyChart"></div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="chart-container">
                            <h4>Feature Importance</h4>
                            <div id="featureChart"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Data will be injected here
        const dashboardData = {DATA_PLACEHOLDER};
        
        function initializeDashboard() {
            updateMetrics();
            createTimelineChart();
            createLevelChart();
            createScoreChart();
            createAnomaliesTable();
            createHourlyChart();
            createFeatureChart();
            updateRecentAlerts();
        }
        
        function updateMetrics() {
            const data = dashboardData;
            document.getElementById('totalLogs').textContent = data.total_logs;
            document.getElementById('totalAnomalies').textContent = data.total_anomalies;
            document.getElementById('anomalyRate').textContent = (data.anomaly_rate * 100).toFixed(1) + '%';
        }
        
        function createTimelineChart() {
            const trace = {
                x: dashboardData.timeline.timestamps,
                y: dashboardData.timeline.anomaly_counts,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Anomalies',
                line: { color: '#dc3545', width: 3 },
                marker: { size: 8 }
            };
            
            const layout = {
                title: 'Anomalies Over Time',
                xaxis: { title: 'Time' },
                yaxis: { title: 'Anomaly Count' },
                plot_bgcolor: '#f8f9fa'
            };
            
            Plotly.newPlot('timelineChart', [trace], layout, { responsive: true });
        }
        
        function createLevelChart() {
            const trace = {
                labels: dashboardData.level_distribution.labels,
                values: dashboardData.level_distribution.values,
                type: 'pie',
                hole: 0.4,
                marker: {
                    colors: ['#28a745', '#ffc107', '#fd7e14', '#dc3545', '#6f42c1']
                }
            };
            
            const layout = {
                title: 'Log Level Distribution',
                showlegend: true,
                plot_bgcolor: '#f8f9fa'
            };
            
            Plotly.newPlot('levelChart', [trace], layout, { responsive: true });
        }
        
        function createScoreChart() {
            const trace = {
                x: dashboardData.score_distribution.scores,
                type: 'histogram',
                nbinsx: 20,
                marker: { color: '#007bff', opacity: 0.7 }
            };
            
            const layout = {
                title: 'Anomaly Score Distribution',
                xaxis: { title: 'Anomaly Score' },
                yaxis: { title: 'Count' },
                plot_bgcolor: '#f8f9fa'
            };
            
            Plotly.newPlot('scoreChart', [trace], layout, { responsive: true });
        }
        
        function createAnomaliesTable() {
            const tableHtml = `
                <table class="table table-striped">
                    <thead>
                        <tr>
                            <th>Timestamp</th>
                            <th>Level</th>
                            <th>Message</th>
                            <th>Score</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${dashboardData.top_anomalies.map(anomaly => `
                            <tr>
                                <td>${anomaly.timestamp}</td>
                                <td><span class="badge bg-${getLevelColor(anomaly.level)}">${anomaly.level}</span></td>
                                <td>${anomaly.message.substring(0, 100)}...</td>
                                <td><span class="badge bg-danger">${anomaly.score.toFixed(3)}</span></td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            `;
            document.getElementById('anomaliesTable').innerHTML = tableHtml;
        }
        
        function createHourlyChart() {
            const trace = {
                x: dashboardData.hourly_pattern.hours,
                y: dashboardData.hourly_pattern.counts,
                type: 'bar',
                marker: { color: '#17a2b8' }
            };
            
            const layout = {
                title: 'Hourly Anomaly Pattern',
                xaxis: { title: 'Hour of Day' },
                yaxis: { title: 'Anomaly Count' },
                plot_bgcolor: '#f8f9fa'
            };
            
            Plotly.newPlot('hourlyChart', [trace], layout, { responsive: true });
        }
        
        function createFeatureChart() {
            const trace = {
                x: Object.values(dashboardData.feature_importance),
                y: Object.keys(dashboardData.feature_importance),
                type: 'bar',
                orientation: 'h',
                marker: { color: '#28a745' }
            };
            
            const layout = {
                title: 'Feature Importance',
                xaxis: { title: 'Importance' },
                plot_bgcolor: '#f8f9fa'
            };
            
            Plotly.newPlot('featureChart', [trace], layout, { responsive: true });
        }
        
        function updateRecentAlerts() {
            const alertsHtml = dashboardData.recent_alerts.map(alert => `
                <div class="alert-item">
                    <small>${alert.timestamp}</small><br>
                    <strong>${alert.message.substring(0, 60)}...</strong><br>
                    <small>Score: ${alert.score.toFixed(3)}</small>
                </div>
            `).join('');
            document.getElementById('recentAlerts').innerHTML = alertsHtml;
        }
        
        function getLevelColor(level) {
            const colors = {
                'ERROR': 'danger',
                'WARNING': 'warning',
                'INFO': 'primary',
                'DEBUG': 'secondary'
            };
            return colors[level] || 'secondary';
        }
        
        function updateDashboard() {
            // Implement filter logic here
            console.log('Updating dashboard with filters...');
        }
        
        function exportData() {
            const dataStr = JSON.stringify(dashboardData, null, 2);
            const dataBlob = new Blob([dataStr], {type: 'application/json'});
            const url = URL.createObjectURL(dataBlob);
            const link = document.createElement('a');
            link.href = url;
            link.download = 'logguard_dashboard_data.json';
            link.click();
        }
        
        function refreshData() {
            location.reload();
        }
        
        // Initialize dashboard when page loads
        document.addEventListener('DOMContentLoaded', initializeDashboard);
    </script>
</body>
</html>
        '''
    
    def _prepare_dashboard_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for the interactive dashboard."""
        anomalies = df[df['anomaly'] == True] if 'anomaly' in df.columns else pd.DataFrame()
        
        # Basic metrics
        total_logs = len(df)
        total_anomalies = len(anomalies)
        anomaly_rate = total_anomalies / total_logs if total_logs > 0 else 0
        
        # Timeline data
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            timeline_data = df.set_index('timestamp').resample('H')['anomaly'].sum()
            timeline = {
                'timestamps': timeline_data.index.strftime('%Y-%m-%d %H:%M').tolist(),
                'anomaly_counts': timeline_data.values.tolist()
            }
        else:
            timeline = {'timestamps': [], 'anomaly_counts': []}
        
        # Level distribution
        level_counts = df['level'].value_counts()
        level_distribution = {
            'labels': level_counts.index.tolist(),
            'values': level_counts.values.tolist()
        }
        
        # Score distribution
        if 'anomaly_score' in df.columns:
            scores = df['anomaly_score'].dropna().values.tolist()
        else:
            scores = []
        
        # Top anomalies
        if len(anomalies) > 0:
            top_anomalies = anomalies.nlargest(10, 'anomaly_score' if 'anomaly_score' in anomalies.columns else 'timestamp')
            top_anomalies_data = []
            for _, row in top_anomalies.iterrows():
                top_anomalies_data.append({
                    'timestamp': str(row.get('timestamp', 'N/A')),
                    'level': str(row.get('level', 'INFO')),
                    'message': str(row.get('message', '')),
                    'score': float(row.get('anomaly_score', 0.5))
                })
        else:
            top_anomalies_data = []
        
        # Hourly pattern
        if 'timestamp' in df.columns and len(anomalies) > 0:
            hourly_pattern = anomalies.groupby(anomalies['timestamp'].dt.hour).size()
            hourly_data = {
                'hours': list(range(24)),
                'counts': [hourly_pattern.get(h, 0) for h in range(24)]
            }
        else:
            hourly_data = {'hours': list(range(24)), 'counts': [0] * 24}
        
        # Feature importance (mock data if not available)
        feature_importance = {
            'message_length': 0.25,
            'log_level': 0.20,
            'timestamp_pattern': 0.15,
            'text_complexity': 0.15,
            'frequency_pattern': 0.10,
            'context_similarity': 0.15
        }
        
        # Recent alerts (last 5 anomalies)
        recent_alerts = top_anomalies_data[:5] if top_anomalies_data else []
        
        return {
            'total_logs': total_logs,
            'total_anomalies': total_anomalies,
            'anomaly_rate': anomaly_rate,
            'timeline': timeline,
            'level_distribution': level_distribution,
            'score_distribution': {'scores': scores},
            'top_anomalies': top_anomalies_data,
            'hourly_pattern': hourly_data,
            'feature_importance': feature_importance,
            'recent_alerts': recent_alerts
        }
    
    def generate_output(self, df: pd.DataFrame, output_path: str, **kwargs) -> None:
        """Generate interactive dashboard HTML."""
        if not self.validate_output_path(output_path):
            raise ValueError(f"Invalid output path: {output_path}")
        
        # Prepare dashboard data
        dashboard_data = self._prepare_dashboard_data(df)
        
        # Generate HTML template
        html_template = self._generate_dashboard_template()
        
        # Replace data placeholder
        html_content = html_template.replace(
            '{DATA_PLACEHOLDER}',
            json.dumps(dashboard_data, indent=2)
        )
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)


class TimeSeriesFormat(OutputFormatPlugin):
    """
    Time-series specific format with advanced temporal analysis.
    
    Generates reports optimized for time-series log analysis with
    trend detection, seasonality analysis, and forecasting.
    """
    
    @property
    def name(self) -> str:
        return "timeseries_format"
    
    @property
    def version(self) -> str:
        return "1.5.0"
    
    @property
    def description(self) -> str:
        return "Time-series analysis format with trend and seasonality detection"
    
    @property
    def file_extension(self) -> str:
        return "json"
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal patterns in the data."""
        if 'timestamp' not in df.columns:
            return {}
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp']).sort_values('timestamp')
        
        analysis = {}
        
        # Time range
        analysis['time_range'] = {
            'start': df['timestamp'].min().isoformat(),
            'end': df['timestamp'].max().isoformat(),
            'duration_hours': (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
        }
        
        # Log frequency analysis
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        
        # Hourly patterns
        hourly_counts = df.groupby('hour').size()
        analysis['hourly_pattern'] = {
            'peak_hour': int(hourly_counts.idxmax()),
            'peak_count': int(hourly_counts.max()),
            'quiet_hour': int(hourly_counts.idxmin()),
            'quiet_count': int(hourly_counts.min()),
            'distribution': hourly_counts.to_dict()
        }
        
        # Daily patterns
        daily_counts = df.groupby('day_of_week').size()
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        analysis['daily_pattern'] = {
            'peak_day': day_names[daily_counts.idxmax()],
            'peak_count': int(daily_counts.max()),
            'quiet_day': day_names[daily_counts.idxmin()],
            'quiet_count': int(daily_counts.min()),
            'distribution': {day_names[i]: int(count) for i, count in daily_counts.items()}
        }
        
        # Anomaly patterns
        if 'anomaly' in df.columns:
            anomalies = df[df['anomaly'] == True]
            if len(anomalies) > 0:
                anomaly_hourly = anomalies.groupby('hour').size()
                anomaly_daily = anomalies.groupby('day_of_week').size()
                
                analysis['anomaly_patterns'] = {
                    'peak_anomaly_hour': int(anomaly_hourly.idxmax()) if len(anomaly_hourly) > 0 else None,
                    'peak_anomaly_day': day_names[anomaly_daily.idxmax()] if len(anomaly_daily) > 0 else None,
                    'hourly_distribution': anomaly_hourly.to_dict(),
                    'daily_distribution': {day_names[i]: int(count) for i, count in anomaly_daily.items()}
                }
        
        # Trend analysis (simple linear trend)
        df['timestamp_numeric'] = df['timestamp'].astype(int) / 10**9  # Convert to seconds
        daily_log_counts = df.groupby(df['timestamp'].dt.date).size()
        
        if len(daily_log_counts) > 1:
            x = np.arange(len(daily_log_counts))
            y = daily_log_counts.values
            
            # Simple linear regression
            slope = np.polyfit(x, y, 1)[0]
            analysis['trend'] = {
                'direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable',
                'slope': float(slope),
                'description': f"Log volume trend: {slope:.2f} logs/day change"
            }
        
        return analysis
    
    def _detect_seasonal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect seasonal patterns in the data."""
        if 'timestamp' not in df.columns or len(df) < 48:  # Need at least 2 days of data
            return {}
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp']).sort_values('timestamp')
        
        # Hourly seasonality
        hourly_pattern = df.groupby(df['timestamp'].dt.hour).size()
        hourly_mean = hourly_pattern.mean()
        hourly_std = hourly_pattern.std()
        
        # Weekly seasonality
        weekly_pattern = df.groupby(df['timestamp'].dt.dayofweek).size()
        weekly_mean = weekly_pattern.mean()
        weekly_std = weekly_pattern.std()
        
        seasonality = {
            'hourly_seasonality': {
                'detected': hourly_std > hourly_mean * 0.1,  # Simple threshold
                'strength': float(hourly_std / hourly_mean) if hourly_mean > 0 else 0,
                'pattern': hourly_pattern.to_dict()
            },
            'weekly_seasonality': {
                'detected': weekly_std > weekly_mean * 0.1,
                'strength': float(weekly_std / weekly_mean) if weekly_mean > 0 else 0,
                'pattern': weekly_pattern.to_dict()
            }
        }
        
        return seasonality
    
    def generate_output(self, df: pd.DataFrame, output_path: str, **kwargs) -> None:
        """Generate time-series analysis JSON report."""
        if not self.validate_output_path(output_path):
            raise ValueError(f"Invalid output path: {output_path}")
        
        # Perform temporal analysis
        temporal_analysis = self._analyze_temporal_patterns(df)
        seasonal_patterns = self._detect_seasonal_patterns(df)
        
        # Compile report
        report = {
            'metadata': {
                'report_type': 'time_series_analysis',
                'generated_at': datetime.now().isoformat(),
                'total_records': len(df),
                'anomaly_count': len(df[df['anomaly'] == True]) if 'anomaly' in df.columns else 0,
                'version': self.version
            },
            'temporal_analysis': temporal_analysis,
            'seasonal_patterns': seasonal_patterns,
            'summary': {
                'key_findings': self._generate_key_findings(temporal_analysis, seasonal_patterns),
                'recommendations': self._generate_recommendations(temporal_analysis, seasonal_patterns)
            }
        }
        
        # Add raw data if requested
        if kwargs.get('include_raw_data', False):
            report['raw_data'] = df.to_dict('records')
        
        # Write JSON report
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
    
    def _generate_key_findings(self, temporal: Dict, seasonal: Dict) -> List[str]:
        """Generate key findings from the analysis."""
        findings = []
        
        if temporal.get('hourly_pattern'):
            peak_hour = temporal['hourly_pattern']['peak_hour']
            findings.append(f"Peak activity occurs at {peak_hour}:00 hours")
        
        if temporal.get('daily_pattern'):
            peak_day = temporal['daily_pattern']['peak_day']
            findings.append(f"Highest log volume on {peak_day}")
        
        if temporal.get('trend'):
            trend = temporal['trend']['direction']
            findings.append(f"Overall trend: {trend}")
        
        if seasonal.get('hourly_seasonality', {}).get('detected'):
            findings.append("Strong hourly seasonality detected")
        
        if seasonal.get('weekly_seasonality', {}).get('detected'):
            findings.append("Weekly pattern identified")
        
        return findings
    
    def _generate_recommendations(self, temporal: Dict, seasonal: Dict) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        if temporal.get('anomaly_patterns'):
            peak_hour = temporal['anomaly_patterns'].get('peak_anomaly_hour')
            if peak_hour is not None:
                recommendations.append(f"Monitor system closely around {peak_hour}:00 hours")
        
        if seasonal.get('hourly_seasonality', {}).get('strength', 0) > 0.5:
            recommendations.append("Consider time-based alerting thresholds due to strong daily patterns")
        
        if temporal.get('trend', {}).get('direction') == 'increasing':
            recommendations.append("Log volume is increasing - consider capacity planning")
        
        recommendations.append("Establish baseline patterns for more accurate anomaly detection")
        
        return recommendations


class ExecutiveSummaryFormat(OutputFormatPlugin):
    """
    Executive summary format for high-level reporting.
    
    Generates concise, business-focused reports suitable for
    executive stakeholders and management reporting.
    """
    
    @property
    def name(self) -> str:
        return "executive_summary"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Executive summary format for management reporting"
    
    @property
    def file_extension(self) -> str:
        return "pdf"  # Would generate PDF in real implementation
    
    def generate_output(self, df: pd.DataFrame, output_path: str, **kwargs) -> None:
        """Generate executive summary report."""
        # For this example, we'll generate an HTML version
        # In a real implementation, this would generate a PDF
        
        output_path = output_path.replace('.pdf', '.html')
        
        if not self.validate_output_path(output_path):
            raise ValueError(f"Invalid output path: {output_path}")
        
        # Calculate key metrics
        total_logs = len(df)
        anomalies = df[df['anomaly'] == True] if 'anomaly' in df.columns else pd.DataFrame()
        anomaly_count = len(anomalies)
        anomaly_rate = (anomaly_count / total_logs * 100) if total_logs > 0 else 0
        
        # Risk assessment
        risk_level = self._assess_risk_level(anomaly_rate)
        
        # Time period
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            time_range = f"{df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}"
        else:
            time_range = "Unknown time period"
        
        # Generate HTML report
        html_content = f'''
<!DOCTYPE html>
<html>
<head>
    <title>Executive Summary - LogGuard ML</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ text-align: center; border-bottom: 2px solid #333; padding-bottom: 20px; }}
        .metric {{ background: #f4f4f4; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .risk-{risk_level.lower()} {{ border-left: 5px solid {"#dc3545" if risk_level == "HIGH" else "#ffc107" if risk_level == "MEDIUM" else "#28a745"}; }}
        .recommendation {{ background: #e9ecef; padding: 15px; margin: 15px 0; border-radius: 5px; }}
        .footer {{ text-align: center; margin-top: 40px; color: #666; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Log Analysis Executive Summary</h1>
        <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
        <p>Analysis Period: {time_range}</p>
    </div>
    
    <h2>Key Metrics</h2>
    <div class="metric">
        <h3>Total Log Entries Analyzed</h3>
        <p style="font-size: 24px; font-weight: bold; color: #007bff;">{total_logs:,}</p>
    </div>
    
    <div class="metric">
        <h3>Anomalies Detected</h3>
        <p style="font-size: 24px; font-weight: bold; color: #dc3545;">{anomaly_count:,}</p>
        <p>({anomaly_rate:.2f}% of total logs)</p>
    </div>
    
    <div class="metric risk-{risk_level.lower()}">
        <h3>Risk Assessment</h3>
        <p style="font-size: 20px; font-weight: bold;">{risk_level} RISK</p>
        <p>{self._get_risk_description(risk_level)}</p>
    </div>
    
    <h2>Impact Analysis</h2>
    {self._generate_impact_analysis(df, anomalies)}
    
    <h2>Recommendations</h2>
    {self._generate_recommendations_html(anomaly_rate, risk_level)}
    
    <div class="footer">
        <p>This report was generated by LogGuard ML v{self.version}</p>
        <p>For detailed technical analysis, please refer to the full technical report.</p>
    </div>
</body>
</html>
        '''
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _assess_risk_level(self, anomaly_rate: float) -> str:
        """Assess risk level based on anomaly rate."""
        if anomaly_rate >= 10:
            return "HIGH"
        elif anomaly_rate >= 5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _get_risk_description(self, risk_level: str) -> str:
        """Get description for risk level."""
        descriptions = {
            "HIGH": "Significant number of anomalies detected. Immediate investigation recommended.",
            "MEDIUM": "Moderate anomaly levels detected. Regular monitoring advised.",
            "LOW": "Low anomaly levels indicate normal system operation."
        }
        return descriptions.get(risk_level, "Risk level assessment unavailable.")
    
    def _generate_impact_analysis(self, df: pd.DataFrame, anomalies: pd.DataFrame) -> str:
        """Generate impact analysis section."""
        if len(anomalies) == 0:
            return "<p>No significant anomalies detected. System appears to be operating normally.</p>"
        
        # Analyze error levels in anomalies
        error_anomalies = len(anomalies[anomalies['level'] == 'ERROR']) if 'level' in anomalies.columns else 0
        warning_anomalies = len(anomalies[anomalies['level'] == 'WARNING']) if 'level' in anomalies.columns else 0
        
        impact_html = f"""
        <div class="metric">
            <h4>Critical Issues</h4>
            <p>{error_anomalies} error-level anomalies detected</p>
        </div>
        <div class="metric">
            <h4>Warning Issues</h4>
            <p>{warning_anomalies} warning-level anomalies detected</p>
        </div>
        """
        
        return impact_html
    
    def _generate_recommendations_html(self, anomaly_rate: float, risk_level: str) -> str:
        """Generate recommendations section."""
        recommendations = []
        
        if risk_level == "HIGH":
            recommendations.extend([
                "Immediate investigation of detected anomalies",
                "Consider implementing additional monitoring",
                "Review system capacity and performance"
            ])
        elif risk_level == "MEDIUM":
            recommendations.extend([
                "Regular monitoring of system logs",
                "Schedule detailed analysis of anomaly patterns",
                "Update alerting thresholds as needed"
            ])
        else:
            recommendations.extend([
                "Continue regular monitoring",
                "Maintain current alerting configuration",
                "Periodic review of detection parameters"
            ])
        
        recommendations.append("Consider implementing automated response procedures")
        
        html = ""
        for rec in recommendations:
            html += f'<div class="recommendation">• {rec}</div>'
        
        return html


class AlertFormat(OutputFormatPlugin):
    """
    Real-time alert format for immediate notifications.
    
    Generates structured alerts suitable for integration with
    monitoring systems, chat platforms, and notification services.
    """
    
    @property
    def name(self) -> str:
        return "alert_format"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Real-time alert format for monitoring integration"
    
    @property
    def file_extension(self) -> str:
        return "json"
    
    def generate_output(self, df: pd.DataFrame, output_path: str, **kwargs) -> None:
        """Generate alert format output."""
        if not self.validate_output_path(output_path):
            raise ValueError(f"Invalid output path: {output_path}")
        
        # Filter for anomalies only
        if 'anomaly' in df.columns:
            anomalies = df[df['anomaly'] == True]
        else:
            anomalies = pd.DataFrame()
        
        # Generate alerts
        alerts = []
        
        for _, row in anomalies.iterrows():
            severity = self._determine_severity(row)
            alert = {
                'id': f"logguard_{hash(str(row.get('message', '')))}",
                'timestamp': str(row.get('timestamp', datetime.now())),
                'severity': severity,
                'title': self._generate_alert_title(row, severity),
                'description': str(row.get('message', 'No message'))[:500],
                'anomaly_score': float(row.get('anomaly_score', 0.5)),
                'log_level': str(row.get('level', 'INFO')),
                'source': 'LogGuard ML',
                'tags': self._generate_tags(row),
                'actions': self._suggest_actions(row, severity)
            }
            alerts.append(alert)
        
        # Create alert summary
        alert_summary = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_alerts': len(alerts),
                'critical_alerts': len([a for a in alerts if a['severity'] == 'CRITICAL']),
                'warning_alerts': len([a for a in alerts if a['severity'] == 'WARNING']),
                'info_alerts': len([a for a in alerts if a['severity'] == 'INFO'])
            },
            'alerts': alerts,
            'summary': self._generate_alert_summary(alerts)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(alert_summary, f, indent=2, default=str)
    
    def _determine_severity(self, row: pd.Series) -> str:
        """Determine alert severity based on log data."""
        level = str(row.get('level', 'INFO')).upper()
        score = float(row.get('anomaly_score', 0.5))
        
        if level == 'ERROR' or score > 0.9:
            return 'CRITICAL'
        elif level == 'WARNING' or score > 0.7:
            return 'WARNING'
        else:
            return 'INFO'
    
    def _generate_alert_title(self, row: pd.Series, severity: str) -> str:
        """Generate alert title."""
        level = str(row.get('level', 'INFO'))
        message = str(row.get('message', ''))
        
        # Extract key information from message
        if 'error' in message.lower():
            return f"{severity}: Error Detected in {level} Log"
        elif 'exception' in message.lower():
            return f"{severity}: Exception Detected"
        elif 'fail' in message.lower():
            return f"{severity}: Failure Detected"
        else:
            return f"{severity}: Anomalous {level} Log Detected"
    
    def _generate_tags(self, row: pd.Series) -> List[str]:
        """Generate tags for the alert."""
        tags = ['logguard', 'anomaly']
        
        level = str(row.get('level', 'INFO')).lower()
        tags.append(f"level:{level}")
        
        message = str(row.get('message', '')).lower()
        if 'database' in message:
            tags.append('database')
        if 'network' in message:
            tags.append('network')
        if 'auth' in message or 'login' in message:
            tags.append('authentication')
        if 'performance' in message:
            tags.append('performance')
        
        return tags
    
    def _suggest_actions(self, row: pd.Series, severity: str) -> List[str]:
        """Suggest actions based on alert."""
        actions = []
        
        if severity == 'CRITICAL':
            actions.extend([
                "Investigate immediately",
                "Check system status",
                "Review recent changes"
            ])
        elif severity == 'WARNING':
            actions.extend([
                "Monitor for patterns",
                "Schedule investigation",
                "Review logs around this time"
            ])
        else:
            actions.extend([
                "Log for future analysis",
                "Check if part of known pattern"
            ])
        
        return actions
    
    def _generate_alert_summary(self, alerts: List[Dict]) -> Dict[str, Any]:
        """Generate summary of all alerts."""
        if not alerts:
            return {"message": "No alerts generated"}
        
        severity_counts = {}
        for alert in alerts:
            severity = alert['severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            "total_alerts": len(alerts),
            "severity_breakdown": severity_counts,
            "most_common_tags": self._get_most_common_tags(alerts),
            "time_span": self._get_time_span(alerts)
        }
    
    def _get_most_common_tags(self, alerts: List[Dict]) -> List[str]:
        """Get most common tags across alerts."""
        tag_counts = {}
        for alert in alerts:
            for tag in alert.get('tags', []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # Return top 5 most common tags
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
        return [tag for tag, count in sorted_tags[:5]]
    
    def _get_time_span(self, alerts: List[Dict]) -> Dict[str, str]:
        """Get time span of alerts."""
        if not alerts:
            return {}
        
        timestamps = [alert['timestamp'] for alert in alerts]
        return {
            "first_alert": min(timestamps),
            "last_alert": max(timestamps)
        }


class ComplianceReportFormat(OutputFormatPlugin):
    """
    Compliance and audit report format.
    
    Generates reports suitable for compliance auditing,
    regulatory requirements, and security assessments.
    """
    
    @property
    def name(self) -> str:
        return "compliance_report"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Compliance and audit report format"
    
    @property
    def file_extension(self) -> str:
        return "xml"
    
    def generate_output(self, df: pd.DataFrame, output_path: str, **kwargs) -> None:
        """Generate compliance report in XML format."""
        if not self.validate_output_path(output_path):
            raise ValueError(f"Invalid output path: {output_path}")
        
        # Create XML structure
        root = ET.Element("ComplianceReport")
        
        # Metadata
        metadata = ET.SubElement(root, "Metadata")
        ET.SubElement(metadata, "GeneratedAt").text = datetime.now().isoformat()
        ET.SubElement(metadata, "ReportVersion").text = self.version
        ET.SubElement(metadata, "Standard").text = kwargs.get('compliance_standard', 'ISO 27001')
        ET.SubElement(metadata, "Auditor").text = kwargs.get('auditor', 'LogGuard ML System')
        
        # Analysis summary
        summary = ET.SubElement(root, "AnalysisSummary")
        ET.SubElement(summary, "TotalLogEntries").text = str(len(df))
        
        anomalies = df[df['anomaly'] == True] if 'anomaly' in df.columns else pd.DataFrame()
        ET.SubElement(summary, "AnomaliesDetected").text = str(len(anomalies))
        ET.SubElement(summary, "AnomalyRate").text = f"{len(anomalies)/len(df)*100:.2f}%" if len(df) > 0 else "0%"
        
        # Security events
        security_events = ET.SubElement(root, "SecurityEvents")
        self._add_security_events(security_events, df, anomalies)
        
        # Compliance findings
        findings = ET.SubElement(root, "ComplianceFindings")
        self._add_compliance_findings(findings, df, anomalies)
        
        # Recommendations
        recommendations = ET.SubElement(root, "Recommendations")
        self._add_recommendations(recommendations, df, anomalies)
        
        # Write XML file
        tree = ET.ElementTree(root)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
    
    def _add_security_events(self, parent: ET.Element, df: pd.DataFrame, anomalies: pd.DataFrame) -> None:
        """Add security-related events to the report."""
        # Authentication failures
        auth_events = df[df['message'].str.contains('auth|login|fail', case=False, na=False)]
        auth_element = ET.SubElement(parent, "AuthenticationEvents")
        ET.SubElement(auth_element, "Count").text = str(len(auth_events))
        
        # Error events
        error_events = df[df['level'] == 'ERROR'] if 'level' in df.columns else pd.DataFrame()
        error_element = ET.SubElement(parent, "ErrorEvents")
        ET.SubElement(error_element, "Count").text = str(len(error_events))
        
        # Anomalous security events
        security_anomalies = anomalies[anomalies['message'].str.contains('security|auth|access', case=False, na=False)]
        security_element = ET.SubElement(parent, "SecurityAnomalies")
        ET.SubElement(security_element, "Count").text = str(len(security_anomalies))
    
    def _add_compliance_findings(self, parent: ET.Element, df: pd.DataFrame, anomalies: pd.DataFrame) -> None:
        """Add compliance findings to the report."""
        findings = []
        
        # Check for high anomaly rates
        anomaly_rate = len(anomalies) / len(df) * 100 if len(df) > 0 else 0
        if anomaly_rate > 10:
            findings.append({
                'severity': 'HIGH',
                'category': 'Monitoring',
                'description': f'High anomaly rate detected: {anomaly_rate:.2f}%',
                'recommendation': 'Investigate root cause and implement additional controls'
            })
        
        # Check for error patterns
        error_count = len(df[df['level'] == 'ERROR']) if 'level' in df.columns else 0
        if error_count > len(df) * 0.05:  # More than 5% errors
            findings.append({
                'severity': 'MEDIUM',
                'category': 'Error Handling',
                'description': f'High error rate detected: {error_count} errors in {len(df)} logs',
                'recommendation': 'Review error handling procedures and implement preventive measures'
            })
        
        # Add findings to XML
        for i, finding in enumerate(findings):
            finding_element = ET.SubElement(parent, f"Finding{i+1}")
            ET.SubElement(finding_element, "Severity").text = finding['severity']
            ET.SubElement(finding_element, "Category").text = finding['category']
            ET.SubElement(finding_element, "Description").text = finding['description']
            ET.SubElement(finding_element, "Recommendation").text = finding['recommendation']
    
    def _add_recommendations(self, parent: ET.Element, df: pd.DataFrame, anomalies: pd.DataFrame) -> None:
        """Add recommendations to the report."""
        recommendations = [
            "Implement continuous monitoring of log anomalies",
            "Establish baseline patterns for normal system behavior",
            "Regular review and update of detection thresholds",
            "Document incident response procedures",
            "Conduct periodic security assessments"
        ]
        
        for i, rec in enumerate(recommendations):
            rec_element = ET.SubElement(parent, f"Recommendation{i+1}")
            rec_element.text = rec
