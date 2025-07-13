"""
HTML Report Generator for LogGuard ML

This module generates HTML reports from processed log data,
highlighting anomalies and providing summary statistics.
"""

import pandas as pd
import os
from datetime import datetime
from typing import Optional

def generate_html_report(df: pd.DataFrame, output_path: str = "reports/anomaly_report.html") -> None:
    """
    Generate an HTML report from the processed log DataFrame.
    
    Args:
        df (pd.DataFrame): Processed log data with anomaly detection results
        output_path (str): Path to save the HTML report
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Calculate summary statistics
    total_logs = len(df)
    anomaly_count = df["is_anomaly"].sum() if "is_anomaly" in df.columns else 0
    error_count = len(df[df["level"] == "ERROR"]) if "level" in df.columns else 0
    warn_count = len(df[df["level"] == "WARN"]) if "level" in df.columns else 0
    info_count = len(df[df["level"] == "INFO"]) if "level" in df.columns else 0
    
    # Get time range
    time_range = "N/A"
    if "timestamp" in df.columns and not df.empty:
        try:
            df_sorted = df.sort_values("timestamp")
            start_time = df_sorted["timestamp"].iloc[0]
            end_time = df_sorted["timestamp"].iloc[-1]
            time_range = f"{start_time} to {end_time}"
        except:
            time_range = "Unable to parse timestamps"
    
    # Generate HTML content
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LogGuard ML - Anomaly Detection Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .summary-card.error {{
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        }}
        .summary-card.warn {{
            background: linear-gradient(135deg, #ffa726 0%, #fb8c00 100%);
        }}
        .summary-card.anomaly {{
            background: linear-gradient(135deg, #e53e3e 0%, #c53030 100%);
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            font-size: 24px;
        }}
        .summary-card p {{
            margin: 0;
            font-size: 14px;
            opacity: 0.9;
        }}
        .table-container {{
            overflow-x: auto;
            margin-top: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
            color: #333;
        }}
        .anomaly-row {{
            background-color: #fff5f5;
            border-left: 4px solid #e53e3e;
        }}
        .error-row {{
            background-color: #fef5e7;
            border-left: 4px solid #dd6b20;
        }}
        .warn-row {{
            background-color: #fffaf0;
            border-left: 4px solid #ed8936;
        }}
        .timestamp {{
            font-family: monospace;
            font-size: 12px;
        }}
        .level {{
            font-weight: bold;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
        }}
        .level-ERROR {{
            background-color: #fed7d7;
            color: #c53030;
        }}
        .level-WARN {{
            background-color: #feebc8;
            color: #dd6b20;
        }}
        .level-INFO {{
            background-color: #e6fffa;
            color: #319795;
        }}
        .footer {{
            margin-top: 30px;
            text-align: center;
            color: #666;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ LogGuard ML - Anomaly Detection Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Time Range:</strong> {time_range}</p>
        </div>
        
        <div class="summary">
            <div class="summary-card">
                <h3>{total_logs}</h3>
                <p>Total Log Entries</p>
            </div>
            <div class="summary-card anomaly">
                <h3>{anomaly_count}</h3>
                <p>Anomalies Detected</p>
            </div>
            <div class="summary-card error">
                <h3>{error_count}</h3>
                <p>Error Messages</p>
            </div>
            <div class="summary-card warn">
                <h3>{warn_count}</h3>
                <p>Warning Messages</p>
            </div>
        </div>
        
        <h2>📊 Log Analysis Details</h2>
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Timestamp</th>
                        <th>Level</th>
                        <th>Message</th>
                        <th>Anomaly</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    # Add table rows
    for _, row in df.iterrows():
        timestamp = row.get("timestamp", "N/A")
        level = row.get("level", "N/A")
        message = row.get("message", "N/A")
        is_anomaly = row.get("is_anomaly", 0)
        
        # Determine row class
        row_class = ""
        if is_anomaly:
            row_class = "anomaly-row"
        elif level == "ERROR":
            row_class = "error-row"
        elif level == "WARN":
            row_class = "warn-row"
        
        # Format level with styling
        level_class = f"level-{level}" if level in ["ERROR", "WARN", "INFO"] else "level"
        level_formatted = f'<span class="level {level_class}">{level}</span>'
        
        # Anomaly indicator
        anomaly_indicator = "🚨 YES" if is_anomaly else "✅ NO"
        
        html_content += f"""
                    <tr class="{row_class}">
                        <td class="timestamp">{timestamp}</td>
                        <td>{level_formatted}</td>
                        <td>{message}</td>
                        <td>{anomaly_indicator}</td>
                    </tr>
"""
    
    # Close HTML
    html_content += f"""
                </tbody>
            </table>
        </div>
        
        <div class="footer">
            <p>Report generated by LogGuard ML | 
               Anomaly detection: {"Enabled" if anomaly_count > 0 or "is_anomaly" in df.columns else "Disabled"}
            </p>
        </div>
    </div>
</body>
</html>
"""
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"[+] HTML report generated: {output_path}")
