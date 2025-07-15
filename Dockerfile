# LogGuard ML - Multi-stage Docker build for production deployment

# Stage 1: Build environment
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install Python dependencies
COPY pyproject.toml README.md ./
COPY logguard_ml/ ./logguard_ml/

# Install the package
RUN pip install --upgrade pip setuptools wheel && \
    pip install .

# Stage 2: Production runtime
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create non-root user for security
RUN groupadd -r logguard && useradd -r -g logguard logguard

# Set up application directories
WORKDIR /app
RUN mkdir -p /app/data /app/config /app/reports /app/logs && \
    chown -R logguard:logguard /app

# Copy application files
COPY --chown=logguard:logguard config/ ./config/
COPY --chown=logguard:logguard data/ ./data/

# Switch to non-root user
USER logguard

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import logguard_ml; print('OK')" || exit 1

# Default command
CMD ["logguard", "--help"]

# Labels for better organization
LABEL maintainer="Saahil Parmar <your.email@example.com>" \
      version="1.0.0" \
      description="LogGuard ML - AI-Powered Log Analysis Framework" \
      org.opencontainers.image.source="https://github.com/SaahilParmar/LogGuard_ML"

# Expose port for web interface (if implemented)
EXPOSE 8080

# Volume mounts for data persistence
VOLUME ["/app/data", "/app/reports", "/app/logs"]
