"""
Advanced Monitoring and Observability for LogGuard ML

This module provides comprehensive monitoring capabilities including:
- Prometheus metrics integration
- Distributed tracing with OpenTelemetry
- Custom dashboards and alerting
- Health checks and SLA monitoring
- Performance telemetry
"""

import logging
import time
import psutil
import threading
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict, deque
import json

# Monitoring dependencies (optional)
try:
    from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("Prometheus client not available. Install with: pip install prometheus-client")

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    # Jaeger exporter is optional
    try:
        from opentelemetry.exporter.jaeger.thrift import JaegerExporter
        JAEGER_AVAILABLE = True
    except ImportError:
        JAEGER_AVAILABLE = False
    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False
    JAEGER_AVAILABLE = False
    logging.warning("OpenTelemetry not available. Install with: pip install opentelemetry-api opentelemetry-sdk")

logger = logging.getLogger(__name__)


@dataclass
class HealthStatus:
    """Health status information."""
    status: str  # 'healthy', 'degraded', 'unhealthy'
    timestamp: datetime
    details: Dict
    uptime: float


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot."""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict
    timestamp: datetime


class PrometheusMetrics:
    """
    Prometheus metrics collection for LogGuard ML.
    """
    
    def __init__(self, port: int = 8000):
        """
        Initialize Prometheus metrics.
        
        Args:
            port: Port to serve metrics on
        """
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus metrics not available")
            return
        
        self.port = port
        self._initialize_metrics()
        self._start_server()
    
    def _initialize_metrics(self):
        """Initialize Prometheus metrics."""
        # Processing metrics
        self.logs_processed_total = Counter(
            'logguard_logs_processed_total',
            'Total number of log entries processed',
            ['status', 'source']
        )
        
        self.processing_duration = Histogram(
            'logguard_processing_duration_seconds',
            'Time spent processing logs',
            ['operation', 'algorithm']
        )
        
        self.anomalies_detected_total = Counter(
            'logguard_anomalies_detected_total',
            'Total number of anomalies detected',
            ['severity', 'algorithm']
        )
        
        # System metrics
        self.memory_usage = Gauge(
            'logguard_memory_usage_bytes',
            'Memory usage in bytes'
        )
        
        self.cpu_usage = Gauge(
            'logguard_cpu_usage_percent',
            'CPU usage percentage'
        )
        
        self.model_accuracy = Gauge(
            'logguard_model_accuracy',
            'Model accuracy score',
            ['algorithm']
        )
        
        # Error metrics
        self.errors_total = Counter(
            'logguard_errors_total',
            'Total number of errors',
            ['error_type', 'component']
        )
        
        # Alert metrics
        self.alerts_sent_total = Counter(
            'logguard_alerts_sent_total',
            'Total number of alerts sent',
            ['channel', 'severity']
        )
    
    def _start_server(self):
        """Start Prometheus metrics server."""
        try:
            start_http_server(self.port)
            logger.info(f"Prometheus metrics server started on port {self.port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")
    
    def record_log_processing(self, count: int, status: str, source: str = "unknown"):
        """Record log processing metrics."""
        if PROMETHEUS_AVAILABLE:
            self.logs_processed_total.labels(status=status, source=source).inc(count)
    
    def record_processing_time(self, operation: str, algorithm: str, duration: float):
        """Record processing time."""
        if PROMETHEUS_AVAILABLE:
            self.processing_duration.labels(operation=operation, algorithm=algorithm).observe(duration)
    
    def record_anomaly_detection(self, count: int, severity: str, algorithm: str):
        """Record anomaly detection."""
        if PROMETHEUS_AVAILABLE:
            self.anomalies_detected_total.labels(severity=severity, algorithm=algorithm).inc(count)
    
    def update_system_metrics(self):
        """Update system resource metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.memory_usage.set(memory.used)
        
        # CPU usage
        cpu_percent = psutil.cpu_percent()
        self.cpu_usage.set(cpu_percent)
    
    def record_error(self, error_type: str, component: str):
        """Record error occurrence."""
        if PROMETHEUS_AVAILABLE:
            self.errors_total.labels(error_type=error_type, component=component).inc()


class DistributedTracing:
    """
    Distributed tracing using OpenTelemetry.
    """
    
    def __init__(self, service_name: str = "logguard-ml", jaeger_endpoint: str = None):
        """
        Initialize distributed tracing.
        
        Args:
            service_name: Name of the service
            jaeger_endpoint: Jaeger collector endpoint
        """
        if not TRACING_AVAILABLE:
            logger.warning("Distributed tracing not available")
            return
        
        self.service_name = service_name
        self._setup_tracing(jaeger_endpoint)
    
    def _setup_tracing(self, jaeger_endpoint: Optional[str]):
        """Setup OpenTelemetry tracing."""
        trace.set_tracer_provider(TracerProvider())
        tracer = trace.get_tracer(__name__)
        
        if jaeger_endpoint and JAEGER_AVAILABLE:
            jaeger_exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=6831,
            )
            span_processor = BatchSpanProcessor(jaeger_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)
            logger.info("Jaeger exporter configured")
        elif jaeger_endpoint and not JAEGER_AVAILABLE:
            logger.warning("Jaeger endpoint specified but Jaeger exporter not available")
        
        self.tracer = tracer
        logger.info(f"Distributed tracing initialized for {self.service_name}")
    
    def trace_operation(self, operation_name: str):
        """
        Decorator for tracing operations.
        
        Args:
            operation_name: Name of the operation to trace
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                if not TRACING_AVAILABLE:
                    return func(*args, **kwargs)
                
                with self.tracer.start_as_current_span(operation_name) as span:
                    span.set_attribute("operation", operation_name)
                    span.set_attribute("service", self.service_name)
                    
                    try:
                        result = func(*args, **kwargs)
                        span.set_attribute("status", "success")
                        return result
                    except Exception as e:
                        span.set_attribute("status", "error")
                        span.set_attribute("error_message", str(e))
                        raise
            return wrapper
        return decorator


class HealthChecker:
    """
    Comprehensive health checking system.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize health checker.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.health_checks = {}
        self.health_history = deque(maxlen=100)
        self.start_time = time.time()
        
        # Register default health checks
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default health checks."""
        self.register_check("memory", self._check_memory)
        self.register_check("cpu", self._check_cpu)
        self.register_check("disk", self._check_disk)
        self.register_check("model_status", self._check_model_status)
    
    def register_check(self, name: str, check_func: Callable[[], Dict]):
        """
        Register a health check function.
        
        Args:
            name: Name of the health check
            check_func: Function that returns health status
        """
        self.health_checks[name] = check_func
        logger.info(f"Registered health check: {name}")
    
    def run_health_checks(self) -> HealthStatus:
        """
        Run all health checks and return overall status.
        
        Returns:
            Overall health status
        """
        check_results = {}
        overall_status = "healthy"
        
        for name, check_func in self.health_checks.items():
            try:
                result = check_func()
                check_results[name] = result
                
                # Determine overall status
                if result.get("status") == "unhealthy":
                    overall_status = "unhealthy"
                elif result.get("status") == "degraded" and overall_status == "healthy":
                    overall_status = "degraded"
                    
            except Exception as e:
                check_results[name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                overall_status = "unhealthy"
        
        health_status = HealthStatus(
            status=overall_status,
            timestamp=datetime.now(),
            details=check_results,
            uptime=time.time() - self.start_time
        )
        
        self.health_history.append(health_status)
        return health_status
    
    def _check_memory(self) -> Dict:
        """Check memory usage."""
        memory = psutil.virtual_memory()
        usage_percent = memory.percent
        
        if usage_percent > 90:
            status = "unhealthy"
        elif usage_percent > 80:
            status = "degraded"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "usage_percent": usage_percent,
            "available_mb": memory.available / (1024 * 1024),
            "total_mb": memory.total / (1024 * 1024)
        }
    
    def _check_cpu(self) -> Dict:
        """Check CPU usage."""
        cpu_percent = psutil.cpu_percent(interval=1)
        
        if cpu_percent > 95:
            status = "unhealthy"
        elif cpu_percent > 85:
            status = "degraded"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "usage_percent": cpu_percent,
            "core_count": psutil.cpu_count()
        }
    
    def _check_disk(self) -> Dict:
        """Check disk usage."""
        disk = psutil.disk_usage('/')
        usage_percent = (disk.used / disk.total) * 100
        
        if usage_percent > 95:
            status = "unhealthy"
        elif usage_percent > 85:
            status = "degraded"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "usage_percent": usage_percent,
            "free_gb": disk.free / (1024**3),
            "total_gb": disk.total / (1024**3)
        }
    
    def _check_model_status(self) -> Dict:
        """Check ML model status."""
        # This would check if models are loaded and responding
        return {
            "status": "healthy",
            "models_loaded": True,
            "last_prediction": datetime.now().isoformat()
        }


class SLAMonitor:
    """
    Service Level Agreement monitoring.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize SLA monitor.
        
        Args:
            config: Configuration with SLA thresholds
        """
        self.config = config
        self.sla_targets = config.get("sla_targets", {
            "availability": 99.9,  # 99.9% uptime
            "response_time_p95": 5.0,  # 5 seconds 95th percentile
            "error_rate": 0.1,  # 0.1% error rate
        })
        
        self.metrics_window = timedelta(hours=24)  # 24-hour rolling window
        self.response_times = deque(maxlen=10000)
        self.error_counts = defaultdict(int)
        self.request_counts = defaultdict(int)
        self.downtime_periods = []
    
    def record_request(self, response_time: float, success: bool, timestamp: datetime = None):
        """
        Record a request for SLA monitoring.
        
        Args:
            response_time: Response time in seconds
            success: Whether the request was successful
            timestamp: Timestamp of the request
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Record response time
        self.response_times.append((response_time, timestamp))
        
        # Record request outcome
        hour_key = timestamp.replace(minute=0, second=0, microsecond=0)
        self.request_counts[hour_key] += 1
        
        if not success:
            self.error_counts[hour_key] += 1
    
    def calculate_sla_metrics(self) -> Dict:
        """
        Calculate current SLA metrics.
        
        Returns:
            Dictionary of SLA metrics
        """
        now = datetime.now()
        window_start = now - self.metrics_window
        
        # Filter recent data
        recent_response_times = [
            rt for rt, ts in self.response_times 
            if ts >= window_start
        ]
        
        # Calculate metrics
        metrics = {}
        
        # Availability
        total_requests = sum(
            count for hour, count in self.request_counts.items()
            if hour >= window_start
        )
        total_errors = sum(
            count for hour, count in self.error_counts.items()
            if hour >= window_start
        )
        
        if total_requests > 0:
            availability = ((total_requests - total_errors) / total_requests) * 100
            error_rate = (total_errors / total_requests) * 100
        else:
            availability = 100.0
            error_rate = 0.0
        
        metrics["availability"] = availability
        metrics["error_rate"] = error_rate
        
        # Response time percentiles
        if recent_response_times:
            recent_response_times.sort()
            p95_index = int(len(recent_response_times) * 0.95)
            p99_index = int(len(recent_response_times) * 0.99)
            
            metrics["response_time_p50"] = recent_response_times[len(recent_response_times) // 2]
            metrics["response_time_p95"] = recent_response_times[p95_index]
            metrics["response_time_p99"] = recent_response_times[p99_index]
            metrics["response_time_avg"] = sum(recent_response_times) / len(recent_response_times)
        else:
            metrics.update({
                "response_time_p50": 0,
                "response_time_p95": 0,
                "response_time_p99": 0,
                "response_time_avg": 0
            })
        
        # SLA compliance
        metrics["sla_compliance"] = {
            "availability": availability >= self.sla_targets["availability"],
            "response_time": metrics["response_time_p95"] <= self.sla_targets["response_time_p95"],
            "error_rate": error_rate <= self.sla_targets["error_rate"]
        }
        
        return metrics


class MonitoringDashboard:
    """
    Real-time monitoring dashboard data provider.
    """
    
    def __init__(self, prometheus_metrics: PrometheusMetrics, health_checker: HealthChecker, sla_monitor: SLAMonitor):
        """
        Initialize monitoring dashboard.
        
        Args:
            prometheus_metrics: Prometheus metrics instance
            health_checker: Health checker instance
            sla_monitor: SLA monitor instance
        """
        self.prometheus_metrics = prometheus_metrics
        self.health_checker = health_checker
        self.sla_monitor = sla_monitor
        
        # Start background monitoring
        self._start_background_monitoring()
    
    def _start_background_monitoring(self):
        """Start background monitoring thread."""
        def monitor_loop():
            while True:
                try:
                    # Update system metrics
                    if self.prometheus_metrics:
                        self.prometheus_metrics.update_system_metrics()
                    
                    time.sleep(30)  # Update every 30 seconds
                except Exception as e:
                    logger.error(f"Monitoring loop error: {e}")
        
        monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitoring_thread.start()
        logger.info("Background monitoring started")
    
    def get_dashboard_data(self) -> Dict:
        """
        Get comprehensive dashboard data.
        
        Returns:
            Dashboard data dictionary
        """
        # Health status
        health_status = self.health_checker.run_health_checks()
        
        # SLA metrics
        sla_metrics = self.sla_monitor.calculate_sla_metrics()
        
        # System metrics
        system_metrics = PerformanceMetrics(
            cpu_usage=psutil.cpu_percent(),
            memory_usage=psutil.virtual_memory().percent,
            disk_usage=psutil.disk_usage('/').percent,
            network_io=psutil.net_io_counters()._asdict(),
            timestamp=datetime.now()
        )
        
        return {
            "timestamp": datetime.now().isoformat(),
            "health": {
                "status": health_status.status,
                "uptime": health_status.uptime,
                "details": health_status.details
            },
            "sla": sla_metrics,
            "system": {
                "cpu_usage": system_metrics.cpu_usage,
                "memory_usage": system_metrics.memory_usage,
                "disk_usage": system_metrics.disk_usage,
                "network_io": system_metrics.network_io
            }
        }


# Integration class for easy setup
class ComprehensiveMonitoring:
    """
    Complete monitoring setup for LogGuard ML.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize comprehensive monitoring.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize components
        self.prometheus_metrics = PrometheusMetrics(
            port=config.get("prometheus_port", 8000)
        ) if PROMETHEUS_AVAILABLE else None
        
        self.tracing = DistributedTracing(
            service_name=config.get("service_name", "logguard-ml"),
            jaeger_endpoint=config.get("jaeger_endpoint")
        ) if TRACING_AVAILABLE else None
        
        self.health_checker = HealthChecker(config)
        self.sla_monitor = SLAMonitor(config)
        
        self.dashboard = MonitoringDashboard(
            self.prometheus_metrics,
            self.health_checker,
            self.sla_monitor
        )
        
        logger.info("Comprehensive monitoring initialized")
    
    def get_monitoring_status(self) -> Dict:
        """Get overall monitoring status."""
        return {
            "prometheus_enabled": PROMETHEUS_AVAILABLE and self.prometheus_metrics is not None,
            "tracing_enabled": TRACING_AVAILABLE and self.tracing is not None,
            "health_checks_active": len(self.health_checker.health_checks),
            "sla_monitoring": True,
            "dashboard_active": True
        }
