"""
Example Custom Output Format Plugins for LogGuard ML

This module demonstrates how to create custom output formats for analysis results.
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import logging

from logguard_ml.plugins import OutputFormatPlugin

logger = logging.getLogger(__name__)


class XMLOutputPlugin(OutputFormatPlugin):
    """XML output format plugin for anomaly detection results."""
    
    @property
    def name(self) -> str:
        return "xml_output"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "XML output format for structured anomaly detection results"
    
    @property
    def file_extension(self) -> str:
        return "xml"
    
    def initialize(self, config: Dict) -> None:
        """Initialize with configuration."""
        xml_config = config.get('xml_output', {})
        self.include_metadata = xml_config.get('include_metadata', True)
        self.pretty_print = xml_config.get('pretty_print', True)
        
    def generate_output(self, df: pd.DataFrame, output_path: str, **kwargs) -> None:
        """Generate XML output from DataFrame."""
        if not self.validate_output_path(output_path):
            raise ValueError(f"Invalid output path: {output_path}")
        
        # Create root element
        root = ET.Element("logguard_results")
        
        # Add metadata if requested
        if self.include_metadata:
            metadata = ET.SubElement(root, "metadata")
            ET.SubElement(metadata, "total_logs").text = str(len(df))
            
            if 'is_anomaly' in df.columns:
                anomaly_count = df['is_anomaly'].sum()
                ET.SubElement(metadata, "anomaly_count").text = str(anomaly_count)
                ET.SubElement(metadata, "anomaly_percentage").text = f"{(anomaly_count/len(df)*100):.2f}%"
            
            # Processing timestamp
            from datetime import datetime
            ET.SubElement(metadata, "generated_at").text = datetime.now().isoformat()
        
        # Add log entries
        logs_element = ET.SubElement(root, "logs")
        
        for idx, row in df.iterrows():
            log_entry = ET.SubElement(logs_element, "log_entry")
            log_entry.set("id", str(idx))
            
            # Add basic fields
            for column in df.columns:
                if column in ['timestamp', 'level', 'message']:
                    element = ET.SubElement(log_entry, column)
                    element.text = str(row[column]) if pd.notna(row[column]) else ""
            
            # Add anomaly information if available
            if 'is_anomaly' in df.columns:
                anomaly_info = ET.SubElement(log_entry, "anomaly_info")
                ET.SubElement(anomaly_info, "is_anomaly").text = str(row['is_anomaly'])
                
                if 'anomaly_score' in df.columns:
                    ET.SubElement(anomaly_info, "score").text = f"{row['anomaly_score']:.4f}"
                
                if 'cluster_id' in df.columns and pd.notna(row['cluster_id']):
                    ET.SubElement(anomaly_info, "cluster_id").text = str(row['cluster_id'])
        
        # Write to file
        if self.pretty_print:
            # Pretty print XML
            rough_string = ET.tostring(root, 'unicode')
            reparsed = minidom.parseString(rough_string)
            pretty_xml = reparsed.toprettyxml(indent="  ")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(pretty_xml)
        else:
            tree = ET.ElementTree(root)
            tree.write(output_path, encoding='utf-8', xml_declaration=True)
        
        logger.info(f"XML output generated: {output_path}")


class JSONLinesOutputPlugin(OutputFormatPlugin):
    """JSON Lines output format plugin (one JSON object per line)."""
    
    @property
    def name(self) -> str:
        return "jsonlines_output"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "JSON Lines output format for streaming and big data processing"
    
    @property
    def file_extension(self) -> str:
        return "jsonl"
    
    def initialize(self, config: Dict) -> None:
        """Initialize with configuration."""
        jsonl_config = config.get('jsonlines_output', {})
        self.include_index = jsonl_config.get('include_index', False)
        self.compact_format = jsonl_config.get('compact_format', True)
        
    def generate_output(self, df: pd.DataFrame, output_path: str, **kwargs) -> None:
        """Generate JSON Lines output from DataFrame."""
        if not self.validate_output_path(output_path):
            raise ValueError(f"Invalid output path: {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for idx, row in df.iterrows():
                # Convert row to dictionary
                record = row.to_dict()
                
                # Add index if requested
                if self.include_index:
                    record['_index'] = idx
                
                # Convert pandas/numpy types to JSON serializable types
                for key, value in record.items():
                    if pd.isna(value):
                        record[key] = None
                    elif hasattr(value, 'item'):  # numpy types
                        record[key] = value.item()
                
                # Write JSON line
                if self.compact_format:
                    json.dump(record, f, separators=(',', ':'))
                else:
                    json.dump(record, f, indent=None)
                f.write('\n')
        
        logger.info(f"JSON Lines output generated: {output_path} ({len(df)} records)")


class MarkdownOutputPlugin(OutputFormatPlugin):
    """Markdown output format plugin for human-readable reports."""
    
    @property
    def name(self) -> str:
        return "markdown_output"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Markdown output format for human-readable anomaly reports"
    
    @property
    def file_extension(self) -> str:
        return "md"
    
    def initialize(self, config: Dict) -> None:
        """Initialize with configuration."""
        md_config = config.get('markdown_output', {})
        self.include_summary = md_config.get('include_summary', True)
        self.max_anomalies_shown = md_config.get('max_anomalies_shown', 50)
        self.include_statistics = md_config.get('include_statistics', True)
        
    def generate_output(self, df: pd.DataFrame, output_path: str, **kwargs) -> None:
        """Generate Markdown output from DataFrame."""
        if not self.validate_output_path(output_path):
            raise ValueError(f"Invalid output path: {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Title
            f.write("# LogGuard ML - Anomaly Detection Report\n\n")
            
            # Timestamp
            from datetime import datetime
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary section
            if self.include_summary:
                f.write("## Summary\n\n")
                f.write(f"- **Total Log Entries:** {len(df)}\n")
                
                if 'is_anomaly' in df.columns:
                    anomaly_count = df['is_anomaly'].sum()
                    f.write(f"- **Anomalies Detected:** {anomaly_count}\n")
                    f.write(f"- **Anomaly Rate:** {(anomaly_count/len(df)*100):.2f}%\n")
                
                f.write("\n")
            
            # Statistics section
            if self.include_statistics and 'is_anomaly' in df.columns:
                f.write("## Statistics\n\n")
                
                # Log level distribution
                if 'level' in df.columns:
                    f.write("### Log Level Distribution\n\n")
                    level_stats = df.groupby('level').agg({
                        'is_anomaly': ['count', 'sum']
                    }).round(2)
                    level_stats.columns = ['Total', 'Anomalies']
                    level_stats['Anomaly Rate %'] = (level_stats['Anomalies'] / level_stats['Total'] * 100).round(2)
                    
                    f.write("| Log Level | Total | Anomalies | Anomaly Rate % |\n")
                    f.write("|-----------|-------|-----------|----------------|\n")
                    
                    for level, row in level_stats.iterrows():
                        f.write(f"| {level} | {row['Total']} | {row['Anomalies']} | {row['Anomaly Rate %']}% |\n")
                    f.write("\n")
                
                # Anomaly score distribution
                if 'anomaly_score' in df.columns:
                    anomaly_scores = df[df['is_anomaly']]['anomaly_score']
                    if len(anomaly_scores) > 0:
                        f.write("### Anomaly Score Statistics\n\n")
                        f.write(f"- **Mean Score:** {anomaly_scores.mean():.4f}\n")
                        f.write(f"- **Median Score:** {anomaly_scores.median():.4f}\n")
                        f.write(f"- **Max Score:** {anomaly_scores.max():.4f}\n")
                        f.write(f"- **Min Score:** {anomaly_scores.min():.4f}\n\n")
            
            # Anomalies section
            if 'is_anomaly' in df.columns:
                anomalies_df = df[df['is_anomaly']].head(self.max_anomalies_shown)
                
                if len(anomalies_df) > 0:
                    f.write(f"## Detected Anomalies\n\n")
                    
                    if len(df[df['is_anomaly']]) > self.max_anomalies_shown:
                        f.write(f"*Showing top {self.max_anomalies_shown} anomalies out of {len(df[df['is_anomaly']])} total*\n\n")
                    
                    for idx, (_, row) in enumerate(anomalies_df.iterrows(), 1):
                        f.write(f"### Anomaly #{idx}\n\n")
                        f.write(f"- **Timestamp:** {row.get('timestamp', 'N/A')}\n")
                        f.write(f"- **Level:** {row.get('level', 'N/A')}\n")
                        f.write(f"- **Message:** `{row.get('message', 'N/A')}`\n")
                        
                        if 'anomaly_score' in row:
                            f.write(f"- **Anomaly Score:** {row['anomaly_score']:.4f}\n")
                        
                        if 'cluster_id' in row and pd.notna(row['cluster_id']):
                            f.write(f"- **Cluster ID:** {row['cluster_id']}\n")
                        
                        f.write("\n")
            
            # Raw data section (optional)
            include_raw = kwargs.get('include_raw_data', False)
            if include_raw:
                f.write("## Raw Data\n\n")
                f.write("```\n")
                f.write(df.to_string())
                f.write("\n```\n")
        
        logger.info(f"Markdown output generated: {output_path}")


class CSVEnhancedOutputPlugin(OutputFormatPlugin):
    """Enhanced CSV output format with additional analysis columns."""
    
    @property
    def name(self) -> str:
        return "csv_enhanced_output"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Enhanced CSV output with additional analysis and metadata columns"
    
    @property
    def file_extension(self) -> str:
        return "csv"
    
    def initialize(self, config: Dict) -> None:
        """Initialize with configuration."""
        csv_config = config.get('csv_enhanced_output', {})
        self.add_analysis_columns = csv_config.get('add_analysis_columns', True)
        self.include_metadata_header = csv_config.get('include_metadata_header', True)
        
    def generate_output(self, df: pd.DataFrame, output_path: str, **kwargs) -> None:
        """Generate enhanced CSV output from DataFrame."""
        if not self.validate_output_path(output_path):
            raise ValueError(f"Invalid output path: {output_path}")
        
        # Create enhanced dataframe
        enhanced_df = df.copy()
        
        if self.add_analysis_columns:
            # Add message length
            enhanced_df['message_length'] = enhanced_df['message'].str.len()
            
            # Add word count
            enhanced_df['word_count'] = enhanced_df['message'].str.split().str.len()
            
            # Add time-based features if timestamp available
            if 'timestamp' in enhanced_df.columns:
                try:
                    timestamps = pd.to_datetime(enhanced_df['timestamp'])
                    enhanced_df['hour'] = timestamps.dt.hour
                    enhanced_df['day_of_week'] = timestamps.dt.day_of_week
                    enhanced_df['is_weekend'] = timestamps.dt.day_of_week.isin([5, 6])
                except:
                    pass
            
            # Add severity score
            severity_mapping = {'DEBUG': 1, 'INFO': 2, 'WARN': 3, 'ERROR': 4}
            if 'level' in enhanced_df.columns:
                enhanced_df['severity_score'] = enhanced_df['level'].map(severity_mapping)
            
            # Add anomaly confidence level
            if 'anomaly_score' in enhanced_df.columns:
                enhanced_df['confidence_level'] = pd.cut(
                    enhanced_df['anomaly_score'],
                    bins=[-float('inf'), 0.3, 0.7, float('inf')],
                    labels=['Low', 'Medium', 'High']
                )
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            # Add metadata header if requested
            if self.include_metadata_header:
                from datetime import datetime
                f.write(f"# LogGuard ML Enhanced CSV Output\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n")
                f.write(f"# Total Records: {len(enhanced_df)}\n")
                
                if 'is_anomaly' in enhanced_df.columns:
                    anomaly_count = enhanced_df['is_anomaly'].sum()
                    f.write(f"# Anomalies: {anomaly_count} ({(anomaly_count/len(enhanced_df)*100):.2f}%)\n")
                
                f.write("# \n")  # Empty comment line for separation
            
            # Write CSV data
            enhanced_df.to_csv(f, index=False, lineterminator='\n')
        
        logger.info(f"Enhanced CSV output generated: {output_path} ({len(enhanced_df)} records)")


class SummaryReportPlugin(OutputFormatPlugin):
    """Executive summary report plugin for high-level insights."""
    
    @property
    def name(self) -> str:
        return "summary_report"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Executive summary report with key insights and recommendations"
    
    @property
    def file_extension(self) -> str:
        return "txt"
    
    def generate_output(self, df: pd.DataFrame, output_path: str, **kwargs) -> None:
        """Generate executive summary report."""
        if not self.validate_output_path(output_path):
            raise ValueError(f"Invalid output path: {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("LOGGUARD ML - EXECUTIVE SUMMARY REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            from datetime import datetime
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Analysis Period: {df['timestamp'].min()} to {df['timestamp'].max()}\n\n" 
                   if 'timestamp' in df.columns else "\n")
            
            # Key metrics
            f.write("KEY METRICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Log Entries Analyzed: {len(df):,}\n")
            
            if 'is_anomaly' in df.columns:
                anomaly_count = df['is_anomaly'].sum()
                f.write(f"Anomalies Detected: {anomaly_count:,}\n")
                f.write(f"Anomaly Rate: {(anomaly_count/len(df)*100):.2f}%\n")
                
                # Risk assessment
                if anomaly_count / len(df) > 0.1:
                    risk_level = "HIGH"
                elif anomaly_count / len(df) > 0.05:
                    risk_level = "MEDIUM"
                else:
                    risk_level = "LOW"
                f.write(f"Risk Level: {risk_level}\n")
            
            f.write("\n")
            
            # Log level analysis
            if 'level' in df.columns:
                f.write("LOG LEVEL ANALYSIS\n")
                f.write("-" * 20 + "\n")
                level_counts = df['level'].value_counts()
                for level, count in level_counts.items():
                    percentage = (count / len(df)) * 100
                    f.write(f"{level}: {count:,} ({percentage:.1f}%)\n")
                f.write("\n")
            
            # Top anomalies
            if 'is_anomaly' in df.columns and 'anomaly_score' in df.columns:
                top_anomalies = df[df['is_anomaly']].nlargest(5, 'anomaly_score')
                if len(top_anomalies) > 0:
                    f.write("TOP ANOMALIES\n")
                    f.write("-" * 20 + "\n")
                    for idx, (_, row) in enumerate(top_anomalies.iterrows(), 1):
                        f.write(f"{idx}. [{row.get('level', 'N/A')}] {row.get('message', 'N/A')[:80]}...\n")
                        f.write(f"   Score: {row['anomaly_score']:.4f}\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 20 + "\n")
            
            if 'is_anomaly' in df.columns:
                anomaly_rate = df['is_anomaly'].sum() / len(df)
                
                if anomaly_rate > 0.1:
                    f.write("• HIGH PRIORITY: Investigate system stability - anomaly rate exceeds 10%\n")
                    f.write("• Review error handling and monitoring systems\n")
                    f.write("• Consider implementing automated alerting\n")
                elif anomaly_rate > 0.05:
                    f.write("• MEDIUM PRIORITY: Monitor system trends\n")
                    f.write("• Review top anomalies for patterns\n")
                else:
                    f.write("• LOW PRIORITY: System appears stable\n")
                    f.write("• Continue routine monitoring\n")
            
            # Error patterns
            if 'level' in df.columns:
                error_rate = len(df[df['level'] == 'ERROR']) / len(df)
                if error_rate > 0.05:
                    f.write("• Review ERROR level logs for recurring issues\n")
            
            f.write("\n")
            f.write("=" * 60 + "\n")
            f.write("End of Report\n")
        
        logger.info(f"Summary report generated: {output_path}")
