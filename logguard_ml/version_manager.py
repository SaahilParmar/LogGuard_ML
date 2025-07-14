"""
Centralized version management for LogGuard ML

This module provides utilities for consistent version handling across the project,
eliminating hardcoded version strings and providing dynamic version access.
"""

import re
from typing import Optional
from pathlib import Path

# Version patterns for validation
VERSION_PATTERN = re.compile(r'^([0-9]+)\.([0-9]+)\.([0-9]+)(?:-([a-zA-Z0-9\-\.]+))?$')


class VersionManager:
    """Centralized version management for LogGuard ML."""
    
    _instance: Optional['VersionManager'] = None
    _version: Optional[str] = None
    
    def __new__(cls) -> 'VersionManager':
        """Singleton pattern to ensure consistent version across the application."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize version manager."""
        if self._version is None:
            self._version = self._load_version()
    
    def _load_version(self) -> str:
        """Load version from __version__.py file."""
        try:
            from logguard_ml.__version__ import __version__
            if self._validate_version(__version__):
                return __version__
            else:
                raise ValueError(f"Invalid version format: {__version__}")
        except ImportError as e:
            raise ImportError(f"Could not import version: {e}")
    
    def _validate_version(self, version: str) -> bool:
        """Validate version format using semantic versioning."""
        return VERSION_PATTERN.match(version) is not None
    
    @property
    def version(self) -> str:
        """Get the current version."""
        return self._version
    
    @property
    def major(self) -> int:
        """Get major version number."""
        return int(self.version.split('.')[0])
    
    @property
    def minor(self) -> int:
        """Get minor version number."""
        return int(self.version.split('.')[1])
    
    @property
    def patch(self) -> int:
        """Get patch version number."""
        return int(self.version.split('.')[2].split('-')[0])
    
    @property
    def prerelease(self) -> Optional[str]:
        """Get prerelease identifier if present."""
        if '-' in self.version:
            return self.version.split('-', 1)[1]
        return None
    
    def is_development(self) -> bool:
        """Check if this is a development version."""
        return self.prerelease is not None
    
    def compare_version(self, other_version: str) -> int:
        """
        Compare current version with another version.
        
        Returns:
            -1 if current version is older
             0 if versions are equal
             1 if current version is newer
        """
        if not self._validate_version(other_version):
            raise ValueError(f"Invalid version format: {other_version}")
        
        current_parts = self._parse_version(self.version)
        other_parts = self._parse_version(other_version)
        
        for i in range(3):  # Compare major, minor, patch
            if current_parts[i] < other_parts[i]:
                return -1
            elif current_parts[i] > other_parts[i]:
                return 1
        
        # Handle prerelease comparison
        current_pre = current_parts[3]
        other_pre = other_parts[3]
        
        if current_pre is None and other_pre is None:
            return 0
        elif current_pre is None and other_pre is not None:
            return 1  # Release > prerelease
        elif current_pre is not None and other_pre is None:
            return -1  # Prerelease < release
        else:
            return -1 if current_pre < other_pre else (1 if current_pre > other_pre else 0)
    
    def _parse_version(self, version: str) -> tuple:
        """Parse version string into components."""
        match = VERSION_PATTERN.match(version)
        if not match:
            raise ValueError(f"Invalid version format: {version}")
        
        major, minor, patch, prerelease = match.groups()
        return (int(major), int(minor), int(patch), prerelease)
    
    def get_version_info(self) -> dict:
        """Get comprehensive version information."""
        return {
            'version': self.version,
            'major': self.major,
            'minor': self.minor,
            'patch': self.patch,
            'prerelease': self.prerelease,
            'is_development': self.is_development(),
            'version_tuple': (self.major, self.minor, self.patch)
        }


# Global version manager instance
version_manager = VersionManager()


def get_version() -> str:
    """Get the current LogGuard ML version."""
    return version_manager.version


def get_version_info() -> dict:
    """Get comprehensive version information."""
    return version_manager.get_version_info()


def validate_version(version: str) -> bool:
    """Validate a version string format."""
    return VERSION_PATTERN.match(version) is not None
