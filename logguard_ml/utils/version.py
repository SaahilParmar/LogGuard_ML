"""
Version utilities for LogGuard ML

Provides centralized version handling, validation, and comparison utilities
to eliminate hardcoded version strings throughout the project.
"""

from logguard_ml.version_manager import (
    get_version, 
    get_version_info, 
    validate_version,
    version_manager
)

__all__ = [
    'get_version',
    'get_version_info', 
    'validate_version',
    'version_manager',
    'get_system_version_info',
    'format_version_display'
]


def get_system_version_info() -> dict:
    """
    Get comprehensive system and version information for reporting.
    
    Returns:
        Dictionary containing version, system, and environment information
    """
    import platform
    import sys
    
    version_info = get_version_info()
    
    return {
        **version_info,
        'python_version': platform.python_version(),
        'python_implementation': platform.python_implementation(),
        'platform': platform.platform(),
        'architecture': platform.machine(),
        'system': platform.system(),
        'release': platform.release(),
        'full_version_string': f"LogGuard ML {get_version()} (Python {platform.python_version()} on {platform.system()})"
    }


def format_version_display(include_prerelease: bool = True) -> str:
    """
    Format version for display purposes.
    
    Args:
        include_prerelease: Whether to include prerelease information
        
    Returns:
        Formatted version string
    """
    version = get_version()
    
    if not include_prerelease and version_manager.is_development():
        return f"{version_manager.major}.{version_manager.minor}.{version_manager.patch}"
    
    return version


def get_api_compatible_version() -> str:
    """
    Get API-compatible version (major.minor only).
    
    Returns:
        API version string (e.g., "1.2")
    """
    return f"{version_manager.major}.{version_manager.minor}"


def is_version_compatible(required_version: str) -> bool:
    """
    Check if current version is compatible with required version.
    
    Args:
        required_version: Minimum required version
        
    Returns:
        True if current version is compatible
    """
    try:
        return version_manager.compare_version(required_version) >= 0
    except ValueError:
        return False
