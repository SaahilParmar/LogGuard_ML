"""
Real-time Log Monitoring Module for LogGuard ML

This module provides real-time log file monitoring capabilities with
streaming anomaly detection and alert generation.

Classes:
    LogMonitor: Real-time log file monitoring with anomaly detection
    AlertManager: Configurable alerting system for anomalies
    StreamProcessor: High-performance streaming log processor

Example:
    >>> from logguard_ml.core.monitoring import LogMonitor
    >>> monitor = LogMonitor(config, log_path="app.log")
    >>> monitor.start_monitoring()
"""

import logging
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from queue import Queue, Empty
from datetime import datetime, timedelta
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import pandas as pd
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .log_parser import LogParser
from .advanced_ml import AdvancedAnomalyDetector
from .performance import PerformanceMonitor, MemoryProfiler

logger = logging.getLogger(__name__)


class LogFileHandler(FileSystemEventHandler):
    """File system event handler for log file monitoring."""
    
    def __init__(self, callback: Callable, target_file: Path):
        """
        Initialize file handler.
        
        Args:
            callback: Function to call when file is modified
            target_file: Path to the log file being monitored
        """
        self.callback = callback
        self.target_file = target_file
        self.last_position = 0
        
        # Initialize position to end of file
        if self.target_file.exists():
            with open(self.target_file, 'r') as f:
                f.seek(0, 2)  # Seek to end
                self.last_position = f.tell()
    
    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory and Path(event.src_path) == self.target_file:
            self.callback()


class AlertManager:
    """Configurable alerting system for anomaly notifications."""
    
    def __init__(self, config: Dict):
        """
        Initialize alert manager.
        
        Args:
            config: Configuration dictionary with alert settings
        """
        self.config = config
        self.alert_config = config.get("alerting", {})
        self.enabled = self.alert_config.get("enabled", False)
        
        # Alert thresholds
        self.anomaly_threshold = self.alert_config.get("anomaly_threshold", 5)
        self.time_window = self.alert_config.get("time_window_minutes", 5)
        
        # Alert channels
        self.email_config = self.alert_config.get("email", {})
        self.webhook_config = self.alert_config.get("webhook", {})
        
        # Alert history for throttling
        self.recent_alerts = []
        self.alert_throttle = self.alert_config.get("throttle_minutes", 15)
        
        logger.info(f"AlertManager initialized: enabled={self.enabled}")
    
    def should_alert(self, anomaly_count: int) -> bool:
        """Determine if an alert should be sent based on thresholds and throttling."""
        if not self.enabled:
            return False
        
        # Check anomaly threshold
        if anomaly_count < self.anomaly_threshold:
            return False
        
        # Check throttling
        now = datetime.now()
        throttle_cutoff = now - timedelta(minutes=self.alert_throttle)
        
        # Remove old alerts
        self.recent_alerts = [alert_time for alert_time in self.recent_alerts if alert_time > throttle_cutoff]
        
        # Check if we should throttle
        if self.recent_alerts:
            logger.debug(f"Alert throttled: {len(self.recent_alerts)} recent alerts")
            return False
        
        return True
    
    def send_alert(self, anomaly_data: Dict):
        """
        Send alert through configured channels.
        
        Args:
            anomaly_data: Dictionary containing anomaly information
        """
        if not self.should_alert(anomaly_data.get("count", 0)):
            return
        
        logger.warning(f"Sending anomaly alert: {anomaly_data['count']} anomalies detected")
        
        # Record alert time
        self.recent_alerts.append(datetime.now())
        
        # Send email alert
        if self.email_config.get("enabled", False):
            self._send_email_alert(anomaly_data)
        
        # Send webhook alert
        if self.webhook_config.get("enabled", False):
            self._send_webhook_alert(anomaly_data)
    
    def _send_email_alert(self, anomaly_data: Dict):
        """Send email alert."""
        try:
            subject = f"LogGuard ML Alert: {anomaly_data['count']} Anomalies Detected"
            body = self._format_alert_message(anomaly_data)
            
            msg = MIMEMultipart()
            msg['From'] = self.email_config['from']
            msg['To'] = ', '.join(self.email_config['to'])
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config.get('smtp_port', 587))
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            server.send_message(msg)
            server.quit()
            
            logger.info("Email alert sent successfully")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def _send_webhook_alert(self, anomaly_data: Dict):
        """Send webhook alert."""
        try:
            import requests
            
            payload = {
                'timestamp': datetime.now().isoformat(),
                'anomaly_count': anomaly_data['count'],
                'details': anomaly_data
            }
            
            response = requests.post(
                self.webhook_config['url'],
                json=payload,
                headers=self.webhook_config.get('headers', {}),
                timeout=10
            )
            
            response.raise_for_status()
            logger.info("Webhook alert sent successfully")
            
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
    
    def _format_alert_message(self, anomaly_data: Dict) -> str:
        """Format alert message as HTML."""
        return f"""
        <html>
        <body>
            <h2>LogGuard ML Anomaly Alert</h2>
            <p><strong>Time:</strong> {anomaly_data.get('timestamp', datetime.now())}</p>
            <p><strong>Anomalies Detected:</strong> {anomaly_data['count']}</p>
            <p><strong>Time Window:</strong> {self.time_window} minutes</p>
            
            <h3>Sample Anomalies:</h3>
            <ul>
            {''.join(f"<li>{msg}</li>" for msg in anomaly_data.get('sample_messages', [])[:5])}
            </ul>
            
            <p>Please review your application logs for potential issues.</p>
        </body>
        </html>
        """


class StreamProcessor:
    """High-performance streaming processor for real-time log analysis."""
    
    def __init__(self, config: Dict, buffer_size: int = 1000, parser: Optional[LogParser] = None, detector: Optional[AdvancedAnomalyDetector] = None):
        """
        Initialize stream processor.
        
        Args:
            config: Configuration dictionary
            buffer_size: Size of processing buffer
            parser: Optional preconfigured log parser
            detector: Optional preconfigured anomaly detector
        """
        self.parser = parser or LogParser(config)
        self.detector = detector or AdvancedAnomalyDetector(config)
        self.buffer_size = buffer_size
        
        # Processing buffers
        self.line_buffer = []
        self.processed_buffer = []
        
        # Performance tracking  
        self.total_processed = 0
        self.anomalies_detected = 0
        self.start_time = time.time()
        
    def process_lines(self, lines: List[str]) -> List[Dict]:
        """
        Process new log lines and detect anomalies.
        
        Args:
            lines: List of new log lines
            
        Returns:
            List of anomaly records
        """
        if not lines:
            return []
        
        # Add to buffer
        self.line_buffer.extend(lines)
        
        # Process buffer if it's large enough
        if len(self.line_buffer) >= self.buffer_size:
            return self._process_buffer()
        
        return []
    
    def flush_buffer(self) -> List[Dict]:
        """Process remaining lines in buffer."""
        if self.line_buffer:
            return self._process_buffer()
        return []
    
    def _process_buffer(self) -> List[Dict]:
        """Process the current buffer and detect anomalies."""
        try:
            # Parse lines into DataFrame
            records = []
            for line_num, line in enumerate(self.line_buffer):
                parsed = self.parser._parse_line(line.strip(), line_num)
                if parsed:
                    records.append(parsed)
            
            if not records:
                self.line_buffer = []
                return []
            
            # Create DataFrame and optimize memory
            df = pd.DataFrame(records)
            df = MemoryProfiler.optimize_dataframe(df)
            
            # Detect anomalies
            df_with_anomalies = self.detector.detect_anomalies(df)
            
            # Extract anomalies
            anomalies = df_with_anomalies[df_with_anomalies['is_anomaly'] == 1]
            
            # Update statistics
            self.total_processed += len(df)
            self.anomalies_detected += len(anomalies)
            
            # Clear buffer
            self.line_buffer = []
            
            # Return anomaly records
            return anomalies.to_dict('records') if not anomalies.empty else []
            
        except Exception as e:
            logger.error(f"Error processing buffer: {e}")
            self.line_buffer = []
            return []
    
    def get_statistics(self) -> Dict:
        """Get processing statistics."""
        runtime = time.time() - self.start_time
        return {
            'total_processed': self.total_processed,
            'anomalies_detected': self.anomalies_detected,
            'anomaly_rate': self.anomalies_detected / max(self.total_processed, 1) * 100,
            'throughput': self.total_processed / max(runtime, 1),
            'runtime_seconds': runtime
        }


class LogMonitor:
    """
    Real-time log monitoring with anomaly detection and alerting.
    
    Monitors log files for new entries, processes them in real-time,
    and sends alerts when anomalies are detected.
    """
    
    def __init__(self, config: Dict, log_path: str):
        """
        Initialize log monitor.
        
        Args:
            config: Configuration dictionary
            log_path: Path to the log file to monitor
        """
        self.config = config
        self.log_path = Path(log_path)
        
        # Initialize components
        self.parser = LogParser(config)
        self.detector = AdvancedAnomalyDetector(config)
        self.alert_manager = AlertManager(config)
        self.stream_processor = StreamProcessor(
            config=config,
            buffer_size=config.get("stream_buffer_size", 100),
            parser=self.parser,
            detector=self.detector
        )
        
        # Monitoring state
        self.is_monitoring = False  # Changed from monitoring to is_monitoring
        self.observer = None
        self.processor_thread = None
        self.line_queue = Queue()
        
        # File position tracking
        self.last_position = 0
        self._initialize_position()
        
        # Performance monitoring
        self.monitor_start_time = None
        
        # Validate file exists
        if not self.log_path.exists():
            raise FileNotFoundError(f"Log file not found: {self.log_path}")
        
        logger.info(f"LogMonitor initialized for {self.log_path}")
    
    def _initialize_position(self):
        """Initialize file position to end of file."""
        if self.log_path.exists():
            with open(self.log_path, 'r', encoding='utf-8', errors='ignore') as f:
                f.seek(0, 2)  # Seek to end
                self.last_position = f.tell()
                logger.debug(f"Initialized file position: {self.last_position}")
    
    def start_monitoring(self):
        """Start real-time log monitoring."""
        if self.is_monitoring:
            logger.warning("Monitoring already started")
            return
        
        logger.info(f"Starting real-time monitoring of {self.log_path}")
        self.is_monitoring = True
        self.monitor_start_time = time.time()
        
        # Start file watcher
        self.observer = Observer()
        handler = LogFileHandler(self._on_file_modified, self.log_path)
        self.observer.schedule(handler, str(self.log_path.parent), recursive=False)
        self.observer.start()
        
        # Start processing thread
        self.processor_thread = threading.Thread(target=self._processor_loop, daemon=True)
        self.processor_thread.start()
        
        logger.info("Real-time monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time log monitoring."""
        if not self.is_monitoring:
            logger.warning("Monitoring not started")
            return
        
        logger.info("Stopping real-time monitoring")
        self.is_monitoring = False
        
        # Stop file watcher
        if self.observer:
            self.observer.stop()
            self.observer.join()
        
        # Process remaining items in queue
        self._process_queue_remaining()
        
        # Log final statistics
        stats = self.stream_processor.get_statistics()
        runtime = time.time() - self.monitor_start_time
        
        logger.info("Monitoring stopped. Final statistics:")
        logger.info(f"  - Runtime: {runtime:.1f}s")
        logger.info(f"  - Total processed: {stats['total_processed']}")
        logger.info(f"  - Anomalies detected: {stats['anomalies_detected']}")
        logger.info(f"  - Anomaly rate: {stats['anomaly_rate']:.2f}%")
        logger.info(f"  - Throughput: {stats['throughput']:.1f} entries/sec")
    
    def read_new_lines(self) -> List[str]:  # Made public
        """Read new lines from the log file."""
        return self._read_new_lines()
    
    def process_anomalies(self, df: pd.DataFrame):  # Added public method
        """Process anomalies in the given DataFrame."""
        if df.empty:
            return
            
        anomalies = df[df['is_anomaly'] == 1]
        if not anomalies.empty:
            anomaly_data = {
                'count': len(anomalies),
                'timestamp': datetime.now().isoformat(),
                'sample_messages': anomalies['message'].tolist()[:5]
            }
            self.alert_manager.send_alert(anomaly_data)
