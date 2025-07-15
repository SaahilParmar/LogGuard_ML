"""
Optimized HTML Report Generator for LogGuard ML

This module generates comprehensive HTML reports from processed log data,
featuring anomaly highlights, interactive visualizations, detailed analytics,
and performance optimizations including caching and memory efficiency.

Functions:
    generate_html_report: Main function to generate HTML reports
    create_summary_stats: Generate summary statistics
    create_visualizations: Create interactive charts and graphs

Example:
    >>> from logguard_ml.reports.report_generator import generate_html_report
    >>> generate_html_report(df, "output/report.html")
"""

import hashlib
import logging
import os
import pickle
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..core.performance import MemoryProfiler, PerformanceMonitor, profile_function

logger = logging.getLogger(__name__)


class ReportGenerationError(Exception):
    """Custom exception for report generation errors."""

    pass


class ReportCache:
    """Caching system for expensive report operations."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize report cache."""
        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "logguard_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.enabled = True

    def _get_cache_key(self, data: pd.DataFrame, operation: str) -> str:
        """Generate cache key for data and operation."""
        # Create hash based on data shape, columns, and operation
        data_hash = hashlib.md5(
            f"{data.shape}_{list(data.columns)}_{operation}".encode()
        ).hexdigest()
        return f"{operation}_{data_hash}"

    def get(self, data: pd.DataFrame, operation: str):
        """Get cached result if available."""
        if not self.enabled:
            return None

        cache_key = self._get_cache_key(data, operation)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    logger.debug(f"Cache hit for {operation}")
                    return pickle.load(f)
        except Exception as e:
            logger.debug(f"Cache read error: {e}")

        return None

    def set(self, data: pd.DataFrame, operation: str, result):
        """Cache the result."""
        if not self.enabled:
            return

        cache_key = self._get_cache_key(data, operation)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)
                logger.debug(f"Cached result for {operation}")
        except Exception as e:
            logger.debug(f"Cache write error: {e}")


# Global cache instance
_report_cache = ReportCache()


@profile_function
def generate_html_report(
    df: pd.DataFrame,
    output_path: str = "reports/anomaly_report.html",
    title: str = "LogGuard ML - Anomaly Detection Report",
    include_raw_data: bool = True,
    use_cache: bool = True,
) -> None:
    """
    Generate a comprehensive HTML report from processed log DataFrame.

    Args:
        df: Processed log data with anomaly detection results
        output_path: Path to save the HTML report
        title: Report title to display
        include_raw_data: Whether to include raw data table in report
        use_cache: Whether to use caching for expensive operations

    Raises:
        ReportGenerationError: If report generation fails
    """
    try:
        logger.info(f"Generating HTML report for {len(df)} log entries")

        # Configure cache
        global _report_cache
        _report_cache.enabled = use_cache

        with PerformanceMonitor() as monitor:
            # Ensure output directory exists
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Optimize DataFrame memory usage
            if not df.empty:
                df = MemoryProfiler.optimize_dataframe(df.copy())

            # Generate report components with caching
            summary_stats = _create_summary_stats(df)
            visualizations = _create_visualizations(df)
            anomaly_table = _create_anomaly_table(df)

            # Build HTML content
            html_content = _build_html_report(
                summary_stats=summary_stats,
                visualizations=visualizations,
                anomaly_table=anomaly_table,
                raw_data_table=_create_raw_data_table(df) if include_raw_data else "",
                title=title,
            )

            # Write to file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            # Log performance statistics
            stats = monitor.get_stats()
            logger.info(f"Report generation completed:")
            logger.info(f"  - Output: {output_path}")
            logger.info(f"  - File size: {output_path.stat().st_size / 1024:.1f} KB")
            logger.info(f"  - Generation time: {stats.execution_time:.2f}s")
            logger.info(f"  - Peak memory: {stats.peak_memory_mb:.1f}MB")

    except Exception as e:
        raise ReportGenerationError(f"Failed to generate report: {e}")


def _create_summary_stats(df: pd.DataFrame) -> Dict:
    """
    Create summary statistics from the DataFrame.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary containing summary statistics
    """
    stats = {
        "total_logs": len(df),
        "anomaly_count": 0,
        "anomaly_percentage": 0.0,
        "error_count": 0,
        "warn_count": 0,
        "info_count": 0,
        "time_range": "N/A",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Anomaly statistics
    if "is_anomaly" in df.columns:
        stats["anomaly_count"] = int(df["is_anomaly"].sum())
        stats["anomaly_percentage"] = (
            (stats["anomaly_count"] / len(df) * 100) if len(df) > 0 else 0
        )

    # Log level statistics
    if "level" in df.columns:
        level_counts = df["level"].value_counts()
        stats["error_count"] = int(level_counts.get("ERROR", 0))
        stats["warn_count"] = int(
            level_counts.get("WARN", 0) + level_counts.get("WARNING", 0)
        )
        stats["info_count"] = int(level_counts.get("INFO", 0))

    # Time range
    if "timestamp" in df.columns and not df.empty:
        try:
            timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
            valid_timestamps = timestamps.dropna()
            if not valid_timestamps.empty:
                start_time = valid_timestamps.min()
                end_time = valid_timestamps.max()
                stats["time_range"] = f"{start_time} to {end_time}"
        except Exception as e:
            logger.warning(f"Could not parse time range: {e}")

    return stats


def _create_visualizations(df: pd.DataFrame) -> str:
    """
    Create interactive visualizations using Plotly.

    Args:
        df: Input DataFrame

    Returns:
        HTML string containing embedded visualizations
    """
    if df.empty:
        return "<p>No data available for visualizations.</p>"

    plots_html = []

    try:
        # 1. Log Level Distribution
        if "level" in df.columns:
            level_fig = _create_level_distribution_chart(df)
            plots_html.append(
                level_fig.to_html(div_id="level-chart", include_plotlyjs=False)
            )

        # 2. Anomaly Timeline
        if "timestamp" in df.columns and "is_anomaly" in df.columns:
            timeline_fig = _create_anomaly_timeline(df)
            if timeline_fig:
                plots_html.append(
                    timeline_fig.to_html(
                        div_id="timeline-chart", include_plotlyjs=False
                    )
                )

        # 3. Message Length Distribution
        if "message" in df.columns:
            length_fig = _create_message_length_distribution(df)
            plots_html.append(
                length_fig.to_html(div_id="length-chart", include_plotlyjs=False)
            )

    except Exception as e:
        logger.warning(f"Error creating visualizations: {e}")
        plots_html.append(f"<p>Error creating visualizations: {e}</p>")

    return "\\n".join(plots_html)


def _create_level_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Create log level distribution pie chart."""
    level_counts = df["level"].value_counts()

    fig = px.pie(
        values=level_counts.values,
        names=level_counts.index,
        title="Log Level Distribution",
        color_discrete_map={
            "ERROR": "#FF6B6B",
            "WARN": "#FFE66D",
            "WARNING": "#FFE66D",
            "INFO": "#4ECDC4",
            "DEBUG": "#95E1D3",
        },
    )

    fig.update_layout(height=400, showlegend=True, title_x=0.5)

    return fig


def _create_anomaly_timeline(df: pd.DataFrame) -> Optional[go.Figure]:
    """Create anomaly timeline chart."""
    try:
        # Convert timestamps
        df_copy = df.copy()
        df_copy["timestamp"] = pd.to_datetime(df_copy["timestamp"], errors="coerce")
        df_copy = df_copy.dropna(subset=["timestamp"])

        if df_copy.empty:
            return None

        # Group by time periods and count anomalies
        df_copy = df_copy.sort_values("timestamp")
        df_copy["hour"] = df_copy["timestamp"].dt.floor("h")

        hourly_stats = (
            df_copy.groupby("hour").agg({"is_anomaly": ["sum", "count"]}).reset_index()
        )

        hourly_stats.columns = ["hour", "anomalies", "total"]

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add total logs line
        fig.add_trace(
            go.Scatter(
                x=hourly_stats["hour"],
                y=hourly_stats["total"],
                mode="lines+markers",
                name="Total Logs",
                line=dict(color="#4ECDC4"),
            ),
            secondary_y=False,
        )

        # Add anomalies bar
        fig.add_trace(
            go.Bar(
                x=hourly_stats["hour"],
                y=hourly_stats["anomalies"],
                name="Anomalies",
                marker_color="#FF6B6B",
                opacity=0.7,
            ),
            secondary_y=True,
        )

        fig.update_xaxes(title_text="Time")
        fig.update_yaxes(title_text="Total Logs", secondary_y=False)
        fig.update_yaxes(title_text="Anomalies", secondary_y=True)

        fig.update_layout(title="Anomaly Timeline", height=400, title_x=0.5)

        return fig

    except Exception as e:
        logger.warning(f"Could not create timeline chart: {e}")
        return None


def _create_message_length_distribution(df: pd.DataFrame) -> go.Figure:
    """Create message length distribution histogram."""
    df_copy = df.copy()
    df_copy["message_length"] = df_copy["message"].astype(str).str.len()

    # Separate normal and anomaly data
    normal_data = df_copy[df_copy.get("is_anomaly", 0) == 0]["message_length"]
    anomaly_data = df_copy[df_copy.get("is_anomaly", 0) == 1]["message_length"]

    fig = go.Figure()

    # Add normal data histogram
    fig.add_trace(
        go.Histogram(
            x=normal_data, name="Normal", opacity=0.7, marker_color="#4ECDC4", nbinsx=30
        )
    )

    # Add anomaly data histogram if available
    if not anomaly_data.empty:
        fig.add_trace(
            go.Histogram(
                x=anomaly_data,
                name="Anomalies",
                opacity=0.7,
                marker_color="#FF6B6B",
                nbinsx=30,
            )
        )

    fig.update_layout(
        title="Message Length Distribution",
        xaxis_title="Message Length (characters)",
        yaxis_title="Frequency",
        barmode="overlay",
        height=400,
        title_x=0.5,
    )

    return fig


def _create_anomaly_table(df: pd.DataFrame) -> str:
    """Create HTML table of anomalous entries."""
    if "is_anomaly" not in df.columns:
        return "<p>No anomaly detection results available.</p>"

    anomalies = df[df["is_anomaly"] == 1]

    if anomalies.empty:
        return "<p>No anomalies detected in the analyzed logs.</p>"

    # Select relevant columns for display
    display_columns = ["timestamp", "level", "message"]
    if "anomaly_score" in anomalies.columns:
        display_columns.append("anomaly_score")

    # Filter to existing columns
    display_columns = [col for col in display_columns if col in anomalies.columns]

    table_html = (
        anomalies[display_columns]
        .head(50)
        .to_html(
            classes="table table-striped table-hover",
            table_id="anomaly-table",
            escape=False,
            index=False,
        )
    )

    if len(anomalies) > 50:
        table_html += f"<p><em>Showing first 50 of {len(anomalies)} anomalies.</em></p>"

    return table_html


def _create_raw_data_table(df: pd.DataFrame) -> str:
    """Create HTML table of raw data (limited to first 100 rows)."""
    if df.empty:
        return "<p>No data available.</p>"

    # Show first 100 rows
    display_df = df.head(100)

    table_html = display_df.to_html(
        classes="table table-striped table-hover table-sm",
        table_id="raw-data-table",
        escape=False,
        index=False,
    )

    if len(df) > 100:
        table_html += f"<p><em>Showing first 100 of {len(df)} total entries.</em></p>"

    return table_html


def _build_html_report(
    summary_stats: Dict,
    visualizations: str,
    anomaly_table: str,
    raw_data_table: str,
    title: str,
) -> str:
    """Build the complete HTML report."""

    # Build raw data section conditionally
    raw_data_section = ""
    if raw_data_table:
        raw_data_section = f"""
        <div class="table-container">
            <h3>Raw Data Sample</h3>
            {raw_data_table}
        </div>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f8f9fa;
        }}
        .navbar {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }}
        .stats-card {{
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}
        .stats-card:hover {{
            transform: translateY(-2px);
        }}
        .anomaly-badge {{
            background: linear-gradient(45deg, #FF6B6B, #FF8E8E);
            color: white;
            border: none;
        }}
        .normal-badge {{
            background: linear-gradient(45deg, #4ECDC4, #6EE7D4);
            color: white;
            border: none;
        }}
        .chart-container {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .table-container {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            max-height: 500px;
            overflow-y: auto;
        }}
        .footer {{
            background: #343a40;
            color: white;
            text-align: center;
            padding: 20px 0;
            margin-top: 50px;
        }}
    </style>
</head>
<body>
    <!-- Navigation -->
    <nav class="navbar navbar-dark">
        <div class="container">
            <span class="navbar-brand mb-0 h1">LogGuard {title}</span>
            <span class="navbar-text">Generated: {summary_stats["generated_at"]}</span>
        </div>
    </nav>

    <div class="container mt-4">
        <!-- Summary Statistics -->
        <div class="row mb-4">
            <div class="col-md-3 mb-3">
                <div class="card stats-card h-100">
                    <div class="card-body text-center">
                        <h5 class="card-title">Total Logs</h5>
                        <h2 class="text-primary">{summary_stats["total_logs"]:,}</h2>
                    </div>
                </div>
            </div>
            <div class="col-md-3 mb-3">
                <div class="card stats-card h-100">
                    <div class="card-body text-center">
                        <h5 class="card-title">Anomalies</h5>
                        <h2 class="text-danger">{summary_stats["anomaly_count"]:,}</h2>
                        <small class="anomaly-badge badge rounded-pill">
                            {summary_stats["anomaly_percentage"]:.2f}%
                        </small>
                    </div>
                </div>
            </div>
            <div class="col-md-3 mb-3">
                <div class="card stats-card h-100">
                    <div class="card-body text-center">
                        <h5 class="card-title">Errors</h5>
                        <h2 class="text-warning">{summary_stats["error_count"]:,}</h2>
                    </div>
                </div>
            </div>
            <div class="col-md-3 mb-3">
                <div class="card stats-card h-100">
                    <div class="card-body text-center">
                        <h5 class="card-title">Warnings</h5>
                        <h2 class="text-info">{summary_stats["warn_count"]:,}</h2>
                    </div>
                </div>
            </div>
        </div>

        <!-- Time Range -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card stats-card">
                    <div class="card-body">
                        <h5 class="card-title">Analysis Period</h5>
                        <p class="card-text">{summary_stats["time_range"]}</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Visualizations -->
        <div class="chart-container">
            <h3>Analytics Dashboard</h3>
            {visualizations}
        </div>

        <!-- Anomaly Details -->
        <div class="table-container">
            <h3>Detected Anomalies</h3>
            {anomaly_table}
        </div>

        <!-- Raw Data -->
        {raw_data_section}
    </div>

    <!-- Footer -->
    <footer class="footer">
        <div class="container">
            <p>&copy; 2025 LogGuard ML - AI-Powered Log Analysis Framework</p>
            <p><small>Generated with LogGuard ML v0.1.0</small></p>
        </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>"""
