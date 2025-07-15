"""
Security Configuration and Input Validation Module

This module provides security enhancements for LogGuard ML including:
- Input sanitization and validation
- Configuration encryption
- Rate limiting
- Audit logging
- Security policy enforcement
"""

import hashlib
import hmac
import json
import logging
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

# Optional cryptography import
try:
    from cryptography.fernet import Fernet

    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    logging.warning(
        "Cryptography not available. Install with: pip install cryptography"
    )

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Custom exception for security-related errors."""

    pass


class InputValidator:
    """
    Validates and sanitizes input data to prevent injection attacks
    and ensure data integrity.
    """

    # Safe regex patterns for log parsing
    SAFE_REGEX_PATTERNS = {
        "timestamp": r"^\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}$",
        "log_level": r"^(DEBUG|INFO|WARN|WARNING|ERROR|FATAL|TRACE)$",
        "ip_address": r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$",
        "alphanumeric": r"^[a-zA-Z0-9_\-\s]*$",
    }

    # Maximum allowed sizes
    MAX_LOG_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    MAX_REGEX_LENGTH = 500
    MAX_MESSAGE_LENGTH = 10000

    @classmethod
    def validate_log_pattern(cls, pattern: str) -> bool:
        """
        Validate regex pattern for safety and complexity.

        Args:
            pattern: Regex pattern to validate

        Returns:
            True if pattern is safe, False otherwise
        """
        if len(pattern) > cls.MAX_REGEX_LENGTH:
            raise SecurityError(
                f"Regex pattern too long: {len(pattern)} > {cls.MAX_REGEX_LENGTH}"
            )

        # Check for potentially dangerous regex constructs
        dangerous_patterns = [
            r"\(\?\#",  # Comments (potential code injection)
            r"\(\?\!",  # Negative lookahead (complexity attack)
            r"\(\?\<\!",  # Negative lookbehind (complexity attack)
            r"\*\+",  # Catastrophic backtracking
            r"\+\*",  # Catastrophic backtracking
        ]

        for dangerous in dangerous_patterns:
            if re.search(dangerous, pattern):
                raise SecurityError(
                    f"Potentially dangerous regex pattern detected: {dangerous}"
                )

        # Test pattern compilation
        try:
            compiled = re.compile(pattern)
            # Test with a simple string to check for excessive complexity
            test_start = time.time()
            compiled.search("test string")
            if time.time() - test_start > 0.1:  # 100ms max
                raise SecurityError("Regex pattern too complex")
        except re.error as e:
            raise SecurityError(f"Invalid regex pattern: {e}")

        return True

    @classmethod
    def sanitize_log_message(cls, message: str) -> str:
        """
        Sanitize log message to prevent log injection attacks.

        Args:
            message: Raw log message

        Returns:
            Sanitized message
        """
        if len(message) > cls.MAX_MESSAGE_LENGTH:
            message = message[: cls.MAX_MESSAGE_LENGTH] + "...[TRUNCATED]"

        # Remove control characters except newlines and tabs
        sanitized = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", message)

        # Escape special characters that could be used for injection
        sanitized = sanitized.replace("\r\n", "\n").replace("\r", "\n")

        return sanitized

    @classmethod
    def validate_file_path(cls, filepath: Union[str, Path]) -> Path:
        """
        Validate file path for security (prevent directory traversal).

        Args:
            filepath: File path to validate

        Returns:
            Validated Path object
        """
        path = Path(filepath).resolve()

        # Check for directory traversal attempts
        if ".." in str(path) or str(path).startswith("/"):
            if not str(path).startswith(str(Path.cwd())):
                raise SecurityError(f"Directory traversal attempt detected: {path}")

        # Check file size
        if path.exists() and path.stat().st_size > cls.MAX_LOG_FILE_SIZE:
            raise SecurityError(
                f"File too large: {path.stat().st_size} > {cls.MAX_LOG_FILE_SIZE}"
            )

        return path


class ConfigurationEncryption:
    """
    Handles encryption and decryption of sensitive configuration data.
    """

    def __init__(self, key_file: Optional[str] = None):
        """
        Initialize encryption handler.

        Args:
            key_file: Path to encryption key file
        """
        self.key_file = key_file or ".logguard_key"
        self.cipher = self._get_or_create_cipher()

    def _get_or_create_cipher(self):
        """Get or create encryption cipher."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ImportError("Cryptography library required for encryption features")

        key_path = Path(self.key_file)

        if key_path.exists():
            with open(key_path, "rb") as f:
                key = f.read()
        else:
            key = Fernet.generate_key()
            with open(key_path, "wb") as f:
                f.write(key)
            # Set restrictive permissions
            key_path.chmod(0o600)

        return Fernet(key)

    def encrypt_value(self, value: str) -> str:
        """Encrypt a configuration value."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ImportError("Cryptography library required for encryption")
        return self.cipher.encrypt(value.encode()).decode()

    def decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt a configuration value."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ImportError("Cryptography library required for decryption")
        return self.cipher.decrypt(encrypted_value.encode()).decode()

    def encrypt_config(self, config: Dict, sensitive_keys: List[str]) -> Dict:
        """
        Encrypt sensitive configuration values.

        Args:
            config: Configuration dictionary
            sensitive_keys: List of keys to encrypt

        Returns:
            Configuration with encrypted values
        """
        encrypted_config = config.copy()

        for key in sensitive_keys:
            if key in config:
                encrypted_config[key] = self.encrypt_value(str(config[key]))
                encrypted_config[f"{key}_encrypted"] = True

        return encrypted_config


class RateLimiter:
    """
    Implements rate limiting for API endpoints and processing requests.
    """

    def __init__(self, max_requests: int = 100, window_minutes: int = 1):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum requests per window
            window_minutes: Time window in minutes
        """
        self.max_requests = max_requests
        self.window = timedelta(minutes=window_minutes)
        self.requests = {}

    def is_allowed(self, client_id: str) -> bool:
        """
        Check if request is allowed for client.

        Args:
            client_id: Unique client identifier

        Returns:
            True if request is allowed
        """
        now = datetime.now()

        if client_id not in self.requests:
            self.requests[client_id] = []

        # Clean old requests
        self.requests[client_id] = [
            req_time
            for req_time in self.requests[client_id]
            if now - req_time < self.window
        ]

        # Check if limit exceeded
        if len(self.requests[client_id]) >= self.max_requests:
            return False

        # Add current request
        self.requests[client_id].append(now)
        return True


class AuditLogger:
    """
    Provides security audit logging capabilities.
    """

    def __init__(self, log_file: str = "security_audit.log"):
        """
        Initialize audit logger.

        Args:
            log_file: Path to audit log file
        """
        self.log_file = log_file
        self.logger = logging.getLogger("security_audit")

        # Configure file handler with restricted permissions
        handler = logging.FileHandler(log_file)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        # Set restrictive file permissions
        Path(log_file).chmod(0o600)

    def log_access(self, user: str, resource: str, action: str, success: bool):
        """Log access attempt."""
        status = "SUCCESS" if success else "FAILURE"
        self.logger.info(
            f"ACCESS {status}: User={user}, Resource={resource}, Action={action}"
        )

    def log_security_event(self, event_type: str, details: Dict):
        """Log security event."""
        self.logger.warning(f"SECURITY EVENT: {event_type} - {json.dumps(details)}")

    def log_configuration_change(self, user: str, changes: Dict):
        """Log configuration changes."""
        self.logger.info(f"CONFIG CHANGE: User={user}, Changes={json.dumps(changes)}")


class SecurityPolicyEnforcer:
    """
    Enforces security policies across the application.
    """

    DEFAULT_POLICIES = {
        "max_file_size": 100 * 1024 * 1024,  # 100MB
        "allowed_file_extensions": [".log", ".txt"],
        "max_processing_time": 300,  # 5 minutes
        "require_encryption": False,
        "audit_all_operations": True,
    }

    def __init__(self, policies: Optional[Dict] = None):
        """
        Initialize security policy enforcer.

        Args:
            policies: Custom security policies
        """
        self.policies = {**self.DEFAULT_POLICIES, **(policies or {})}
        self.audit_logger = AuditLogger()

    def validate_file_upload(self, filepath: Path, user: str) -> bool:
        """
        Validate file upload against security policies.

        Args:
            filepath: Path to uploaded file
            user: User performing upload

        Returns:
            True if upload is allowed
        """
        try:
            # Check file extension
            if filepath.suffix not in self.policies["allowed_file_extensions"]:
                self.audit_logger.log_security_event(
                    "INVALID_FILE_EXTENSION",
                    {"file": str(filepath), "user": user, "extension": filepath.suffix},
                )
                return False

            # Check file size
            if filepath.stat().st_size > self.policies["max_file_size"]:
                self.audit_logger.log_security_event(
                    "FILE_TOO_LARGE",
                    {
                        "file": str(filepath),
                        "user": user,
                        "size": filepath.stat().st_size,
                    },
                )
                return False

            self.audit_logger.log_access(user, str(filepath), "FILE_UPLOAD", True)
            return True

        except Exception as e:
            self.audit_logger.log_security_event(
                "FILE_VALIDATION_ERROR",
                {"file": str(filepath), "user": user, "error": str(e)},
            )
            return False
