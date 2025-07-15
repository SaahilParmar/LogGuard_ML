"""
FastAPI Web Application for LogGuard ML

This module provides a modern web interface and REST API for LogGuard ML including:
- Interactive web dashboard
- REST API endpoints
- Real-time websocket connections
- File upload and management
- User authentication and authorization
"""

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, WebSocket, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
import uvicorn
import asyncio
import logging
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

# LogGuard ML imports
from logguard_ml.core.log_parser import LogParser
from logguard_ml.core.advanced_ml import AdvancedAnomalyDetector
from logguard_ml.core.observability import ComprehensiveMonitoring
from logguard_ml.utils.security import SecurityPolicyEnforcer, InputValidator
from logguard_ml.reports.report_generator import generate_html_report

logger = logging.getLogger(__name__)

# Pydantic models for API
class LogAnalysisRequest(BaseModel):
    """Request model for log analysis."""
    file_path: Optional[str] = None
    log_data: Optional[str] = None
    config_override: Optional[Dict] = None
    use_ml: bool = True
    algorithm: str = "isolation_forest"

class LogAnalysisResponse(BaseModel):
    """Response model for log analysis."""
    analysis_id: str
    status: str
    total_logs: int
    anomalies_detected: int
    processing_time: float
    results: Optional[Dict] = None
    report_url: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: datetime
    uptime: float
    details: Dict

class MetricsResponse(BaseModel):
    """Metrics response model."""
    timestamp: datetime
    system_metrics: Dict
    sla_metrics: Dict
    performance_metrics: Dict

class ConfigUpdateRequest(BaseModel):
    """Configuration update request."""
    log_patterns: Optional[List[Dict]] = None
    ml_model: Optional[Dict] = None
    monitoring: Optional[Dict] = None
    alerting: Optional[Dict] = None

# FastAPI application
app = FastAPI(
    title="LogGuard ML API",
    description="AI-Powered Log Analysis & Anomaly Detection",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Security
security = HTTPBearer()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global state
analysis_jobs = {}
websocket_connections = []
monitoring_system = None
security_enforcer = None


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Validate JWT token and return current user.
    
    Args:
        credentials: HTTP authorization credentials
        
    Returns:
        User information
    """
    # Simplified authentication - implement proper JWT validation in production
    token = credentials.credentials
    
    # For demo purposes, accept any token
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return {"user_id": "demo_user", "username": "demo"}


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    global monitoring_system, security_enforcer
    
    # Initialize monitoring
    monitoring_config = {
        "prometheus_port": 8001,
        "service_name": "logguard-ml-api"
    }
    monitoring_system = ComprehensiveMonitoring(monitoring_config)
    
    # Initialize security
    security_policies = {
        "max_file_size": 50 * 1024 * 1024,  # 50MB
        "allowed_file_extensions": [".log", ".txt"],
        "audit_all_operations": True
    }
    security_enforcer = SecurityPolicyEnforcer(security_policies)
    
    logger.info("LogGuard ML API started successfully")


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the main dashboard."""
    return templates.TemplateResponse("dashboard.html", {"request": {}})


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """
    Get application health status.
    
    Returns:
        Health status information
    """
    if monitoring_system:
        health_status = monitoring_system.health_checker.run_health_checks()
        return HealthResponse(
            status=health_status.status,
            timestamp=health_status.timestamp,
            uptime=health_status.uptime,
            details=health_status.details
        )
    else:
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now(),
            uptime=0.0,
            details={}
        )


@app.get("/api/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Get system and performance metrics.
    
    Returns:
        Comprehensive metrics data
    """
    if monitoring_system:
        dashboard_data = monitoring_system.dashboard.get_dashboard_data()
        return MetricsResponse(
            timestamp=datetime.fromisoformat(dashboard_data["timestamp"]),
            system_metrics=dashboard_data["system"],
            sla_metrics=dashboard_data["sla"],
            performance_metrics={}
        )
    else:
        return MetricsResponse(
            timestamp=datetime.now(),
            system_metrics={},
            sla_metrics={},
            performance_metrics={}
        )


@app.post("/api/analyze", response_model=LogAnalysisResponse)
async def analyze_logs(
    request: LogAnalysisRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Analyze log files for anomalies.
    
    Args:
        request: Log analysis request
        current_user: Current authenticated user
        
    Returns:
        Analysis results
    """
    analysis_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    try:
        # Validate input
        if not request.file_path and not request.log_data:
            raise HTTPException(
                status_code=400,
                detail="Either file_path or log_data must be provided"
            )
        
        # Security validation
        if request.file_path and security_enforcer:
            file_path = Path(request.file_path)
            if not security_enforcer.validate_file_upload(file_path, current_user["username"]):
                raise HTTPException(
                    status_code=403,
                    detail="File upload validation failed"
                )
        
        # Load configuration
        config = {
            "log_patterns": [
                {"pattern": r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (?P<level>ERROR|WARN|INFO) (?P<message>.+)"}
            ],
            "ml_model": {
                "algorithm": request.algorithm,
                "contamination": 0.05,
                "random_state": 42
            }
        }
        
        # Override with user config
        if request.config_override:
            config.update(request.config_override)
        
        # Parse logs
        parser = LogParser(config)
        if request.file_path:
            df = parser.parse_log_file(request.file_path)
        else:
            # Create temporary file for log data
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
                f.write(request.log_data)
                temp_file = f.name
            df = parser.parse_log_file(temp_file)
            Path(temp_file).unlink()  # Clean up
        
        # Detect anomalies if requested
        anomalies_count = 0
        if request.use_ml and not df.empty:
            detector = AdvancedAnomalyDetector(config)
            df_with_anomalies = detector.detect_anomalies(df)
            anomalies_count = len(df_with_anomalies[df_with_anomalies['anomaly'] == -1])
        else:
            df_with_anomalies = df
        
        # Generate report
        report_path = f"reports/analysis_{analysis_id}.html"
        Path("reports").mkdir(exist_ok=True)
        generate_html_report(df_with_anomalies, report_path)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Store job result
        job_result = LogAnalysisResponse(
            analysis_id=analysis_id,
            status="completed",
            total_logs=len(df),
            anomalies_detected=anomalies_count,
            processing_time=processing_time,
            results={
                "anomaly_percentage": (anomalies_count / len(df)) * 100 if len(df) > 0 else 0,
                "algorithm_used": request.algorithm,
                "timestamp": start_time.isoformat()
            },
            report_url=f"/api/reports/{analysis_id}"
        )
        
        analysis_jobs[analysis_id] = job_result
        
        # Record metrics
        if monitoring_system and monitoring_system.prometheus_metrics:
            monitoring_system.prometheus_metrics.record_log_processing(
                len(df), "success", "api"
            )
            monitoring_system.prometheus_metrics.record_processing_time(
                "analysis", request.algorithm, processing_time
            )
            monitoring_system.prometheus_metrics.record_anomaly_detection(
                anomalies_count, "medium", request.algorithm
            )
        
        # Notify websocket clients
        await notify_websocket_clients({
            "type": "analysis_complete",
            "analysis_id": analysis_id,
            "total_logs": len(df),
            "anomalies_detected": anomalies_count
        })
        
        return job_result
        
    except Exception as e:
        logger.error(f"Analysis failed for {analysis_id}: {e}")
        
        # Record error metrics
        if monitoring_system and monitoring_system.prometheus_metrics:
            monitoring_system.prometheus_metrics.record_error("analysis_failed", "api")
        
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    current_user: Dict = Depends(get_current_user)
):
    """
    Upload log file for analysis.
    
    Args:
        file: Uploaded file
        current_user: Current authenticated user
        
    Returns:
        Upload confirmation
    """
    try:
        # Validate file
        if not InputValidator.validate_log_pattern(file.filename):
            raise HTTPException(
                status_code=400,
                detail="Invalid file format"
            )
        
        # Save file
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / f"{uuid.uuid4()}_{file.filename}"
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Security validation
        if security_enforcer:
            if not security_enforcer.validate_file_upload(file_path, current_user["username"]):
                file_path.unlink()  # Remove file
                raise HTTPException(
                    status_code=403,
                    detail="File security validation failed"
                )
        
        return {
            "filename": file.filename,
            "file_path": str(file_path),
            "size": len(content),
            "upload_time": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(e)}"
        )


@app.get("/api/analysis/{analysis_id}")
async def get_analysis_status(analysis_id: str):
    """
    Get analysis job status.
    
    Args:
        analysis_id: Analysis job ID
        
    Returns:
        Analysis status
    """
    if analysis_id not in analysis_jobs:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found"
        )
    
    return analysis_jobs[analysis_id]


@app.get("/api/reports/{analysis_id}")
async def get_analysis_report(analysis_id: str):
    """
    Get analysis report.
    
    Args:
        analysis_id: Analysis job ID
        
    Returns:
        HTML report
    """
    report_path = Path(f"reports/analysis_{analysis_id}.html")
    
    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Report not found"
        )
    
    with open(report_path, "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/config")
async def get_config():
    """Get current configuration."""
    # Return default configuration
    return {
        "log_patterns": [
            {"pattern": r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (?P<level>ERROR|WARN|INFO) (?P<message>.+)"}
        ],
        "ml_model": {
            "algorithm": "isolation_forest",
            "contamination": 0.05,
            "random_state": 42
        },
        "monitoring": {
            "enabled": True,
            "check_interval": 1
        }
    }


@app.post("/api/config")
async def update_config(
    request: ConfigUpdateRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Update configuration.
    
    Args:
        request: Configuration update request
        current_user: Current authenticated user
        
    Returns:
        Updated configuration
    """
    # In production, this would update the actual configuration
    # For now, just return the requested changes
    
    updated_config = {}
    if request.log_patterns:
        updated_config["log_patterns"] = request.log_patterns
    if request.ml_model:
        updated_config["ml_model"] = request.ml_model
    if request.monitoring:
        updated_config["monitoring"] = request.monitoring
    if request.alerting:
        updated_config["alerting"] = request.alerting
    
    # Log configuration change
    if security_enforcer:
        security_enforcer.audit_logger.log_configuration_change(
            current_user["username"], updated_config
        )
    
    return {
        "status": "updated",
        "changes": updated_config,
        "timestamp": datetime.now().isoformat()
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time updates.
    
    Args:
        websocket: WebSocket connection
    """
    await websocket.accept()
    websocket_connections.append(websocket)
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        websocket_connections.remove(websocket)


async def notify_websocket_clients(message: Dict):
    """
    Notify all connected WebSocket clients.
    
    Args:
        message: Message to send
    """
    if websocket_connections:
        message_str = json.dumps(message)
        for websocket in websocket_connections.copy():
            try:
                await websocket.send_text(message_str)
            except Exception as e:
                logger.error(f"Failed to send websocket message: {e}")
                websocket_connections.remove(websocket)


# Development server
if __name__ == "__main__":
    uvicorn.run(
        "web_api:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )
