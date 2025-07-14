"""
Plugin System for LogGuard ML

This module provides a flexible plugin architecture that allows users to extend
LogGuard ML with custom ML algorithms, output formats, and processing capabilities.

Features:
- Dynamic plugin loading and registration
- Custom ML algorithm plugins
- Custom output format plugins  
- Custom log parser plugins
- Plugin lifecycle management
- Configuration validation for plugins

Example:
    >>> from logguard_ml.plugins import PluginManager
    >>> manager = PluginManager()
    >>> manager.load_plugins("plugins/")
    >>> custom_detector = manager.get_detector("my_custom_algorithm")
"""

import importlib
import importlib.util
import inspect
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Type, Any, Optional, Callable
import yaml
import json

logger = logging.getLogger(__name__)


class PluginError(Exception):
    """Base exception for plugin-related errors."""
    pass


class PluginInterface(ABC):
    """Base interface for all LogGuard ML plugins."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name identifier."""
        pass
    
    @property
    @abstractmethod 
    def version(self) -> str:
        """Plugin version."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Plugin description."""
        pass
    
    def initialize(self, config: Dict) -> None:
        """Initialize plugin with configuration."""
        pass
    
    def cleanup(self) -> None:
        """Cleanup plugin resources."""
        pass


class MLDetectorPlugin(PluginInterface):
    """Base class for custom ML anomaly detection algorithms."""
    
    @abstractmethod
    def detect_anomalies(self, df, config: Dict) -> Any:
        """
        Detect anomalies in the provided DataFrame.
        
        Args:
            df: Pandas DataFrame with log data
            config: Configuration dictionary
            
        Returns:
            DataFrame with anomaly predictions
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importance scores."""
        pass
    
    def validate_data(self, df) -> bool:
        """Validate input data format."""
        required_columns = ['message', 'level', 'timestamp']
        return all(col in df.columns for col in required_columns)


class OutputFormatPlugin(PluginInterface):
    """Base class for custom output format plugins."""
    
    @property
    @abstractmethod
    def file_extension(self) -> str:
        """File extension for this format (e.g., 'xml', 'pdf')."""
        pass
    
    @abstractmethod
    def generate_output(self, df, output_path: str, **kwargs) -> None:
        """
        Generate output in custom format.
        
        Args:
            df: DataFrame with anomaly detection results
            output_path: Path for output file
            **kwargs: Additional formatting options
        """
        pass
    
    def validate_output_path(self, output_path: str) -> bool:
        """Validate output path is writable."""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            return True
        except Exception:
            return False


class LogParserPlugin(PluginInterface):
    """Base class for custom log parser plugins."""
    
    @property
    @abstractmethod
    def supported_formats(self) -> List[str]:
        """List of log formats this parser supports."""
        pass
    
    @abstractmethod
    def parse_log_line(self, line: str) -> Dict[str, Any]:
        """
        Parse a single log line.
        
        Args:
            line: Raw log line string
            
        Returns:
            Dictionary with extracted fields
        """
        pass
    
    def can_parse(self, log_sample: str) -> bool:
        """Check if this parser can handle the log format."""
        try:
            lines = log_sample.split('\n')[:5]  # Test first 5 lines
            for line in lines:
                if line.strip():
                    result = self.parse_log_line(line)
                    if not result or not isinstance(result, dict):
                        return False
            return True
        except Exception:
            return False


class PluginMetadata:
    """Metadata container for plugins."""
    
    def __init__(self, 
                 name: str,
                 version: str, 
                 description: str,
                 author: str = "",
                 dependencies: List[str] = None,
                 config_schema: Dict = None):
        self.name = name
        self.version = version
        self.description = description
        self.author = author
        self.dependencies = dependencies or []
        self.config_schema = config_schema or {}
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'PluginMetadata':
        """Create metadata from dictionary."""
        return cls(**data)
        
    def to_dict(self) -> Dict:
        """Convert metadata to dictionary."""
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'author': self.author,
            'dependencies': self.dependencies,
            'config_schema': self.config_schema
        }


class PluginManager:
    """Manager for loading and managing plugins."""
    
    def __init__(self):
        """Initialize plugin manager."""
        self.ml_detectors: Dict[str, Type[MLDetectorPlugin]] = {}
        self.output_formats: Dict[str, Type[OutputFormatPlugin]] = {}
        self.log_parsers: Dict[str, Type[LogParserPlugin]] = {}
        self.plugin_metadata: Dict[str, PluginMetadata] = {}
        self.loaded_plugins: Dict[str, PluginInterface] = {}
        
    def register_ml_detector(self, detector_class: Type[MLDetectorPlugin]) -> None:
        """Register a custom ML detector plugin."""
        if not inspect.isclass(detector_class):
            raise PluginError(f"Expected a class, got {type(detector_class).__name__}")
            
        if not issubclass(detector_class, MLDetectorPlugin):
            raise PluginError(f"Class must inherit from MLDetectorPlugin")
            
        instance = detector_class()
        name = instance.name
        
        self.ml_detectors[name] = detector_class
        self.plugin_metadata[name] = PluginMetadata(
            name=name,
            version=instance.version,
            description=instance.description
        )
        
        logger.info(f"Registered ML detector plugin: {name}")
        
    def register_output_format(self, format_class: Type[OutputFormatPlugin]) -> None:
        """Register a custom output format plugin."""
        if not inspect.isclass(format_class):
            raise PluginError(f"Expected a class, got {type(format_class).__name__}")
            
        if not issubclass(format_class, OutputFormatPlugin):
            raise PluginError(f"Class must inherit from OutputFormatPlugin")
            
        instance = format_class()
        name = instance.name
        
        self.output_formats[name] = format_class
        self.plugin_metadata[name] = PluginMetadata(
            name=name,
            version=instance.version,
            description=instance.description
        )
        
        logger.info(f"Registered output format plugin: {name}")
        
    def register_log_parser(self, parser_class: Type[LogParserPlugin]) -> None:
        """Register a custom log parser plugin."""
        if not inspect.isclass(parser_class):
            raise PluginError(f"Expected a class, got {type(parser_class).__name__}")
            
        if not issubclass(parser_class, LogParserPlugin):
            raise PluginError(f"Class must inherit from LogParserPlugin")
            
        instance = parser_class()
        name = instance.name
        
        self.log_parsers[name] = parser_class
        self.plugin_metadata[name] = PluginMetadata(
            name=name,
            version=instance.version,
            description=instance.description
        )
        
        logger.info(f"Registered log parser plugin: {name}")
        
    def load_plugin_file(self, plugin_path: Path) -> None:
        """Load a single plugin file."""
        try:
            # Read plugin metadata if exists
            metadata_file = plugin_path.parent / f"{plugin_path.stem}_metadata.yaml"
            metadata = None
            
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata_dict = yaml.safe_load(f)
                    metadata = PluginMetadata.from_dict(metadata_dict)
                    
            # Import the plugin module
            spec = importlib.util.spec_from_file_location(plugin_path.stem, plugin_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin classes in the module
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if name.startswith('_'):
                    continue
                    
                if issubclass(obj, MLDetectorPlugin) and obj != MLDetectorPlugin:
                    self.register_ml_detector(obj)
                elif issubclass(obj, OutputFormatPlugin) and obj != OutputFormatPlugin:
                    self.register_output_format(obj)
                elif issubclass(obj, LogParserPlugin) and obj != LogParserPlugin:
                    self.register_log_parser(obj)
                    
            logger.info(f"Loaded plugin: {plugin_path}")
            
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_path}: {e}")
            raise PluginError(f"Failed to load plugin {plugin_path}: {e}")
            
    def load_plugins_from_directory(self, plugins_dir: Path) -> None:
        """Load all plugins from a directory."""
        if not plugins_dir.exists():
            logger.warning(f"Plugin directory does not exist: {plugins_dir}")
            return
            
        for plugin_file in plugins_dir.glob("*.py"):
            if plugin_file.name.startswith("__"):
                continue
            self.load_plugin_file(plugin_file)
            
    def get_ml_detector(self, name: str, config: Dict = None) -> MLDetectorPlugin:
        """Get an instance of a registered ML detector."""
        if name not in self.ml_detectors:
            raise PluginError(f"ML detector '{name}' not found")
            
        detector_class = self.ml_detectors[name]
        instance = detector_class()
        
        if config:
            instance.initialize(config)
            
        return instance
        
    def get_output_format(self, name: str, config: Dict = None) -> OutputFormatPlugin:
        """Get an instance of a registered output format."""
        if name not in self.output_formats:
            raise PluginError(f"Output format '{name}' not found")
            
        format_class = self.output_formats[name]
        instance = format_class()
        
        if config:
            instance.initialize(config)
            
        return instance
        
    def get_log_parser(self, name: str, config: Dict = None) -> LogParserPlugin:
        """Get an instance of a registered log parser."""
        if name not in self.log_parsers:
            raise PluginError(f"Log parser '{name}' not found")
            
        parser_class = self.log_parsers[name]
        instance = parser_class()
        
        if config:
            instance.initialize(config)
            
        return instance
        
    def list_plugins(self) -> Dict[str, List[str]]:
        """List all registered plugins by category."""
        return {
            'ml_detectors': list(self.ml_detectors.keys()),
            'output_formats': list(self.output_formats.keys()),
            'log_parsers': list(self.log_parsers.keys())
        }
        
    def get_plugin_info(self, name: str) -> Dict:
        """Get detailed information about a plugin."""
        if name not in self.plugin_metadata:
            raise PluginError(f"Plugin '{name}' not found")
            
        return self.plugin_metadata[name].to_dict()
        
    def validate_plugin_config(self, plugin_name: str, config: Dict) -> bool:
        """Validate configuration for a plugin."""
        if plugin_name not in self.plugin_metadata:
            return True  # No validation schema available
            
        metadata = self.plugin_metadata[plugin_name]
        if not metadata.config_schema:
            return True
            
        # TODO: Implement JSON schema validation
        return True
        
    def export_plugin_registry(self, output_path: str) -> None:
        """Export plugin registry information."""
        registry_data = {
            'plugins': {name: meta.to_dict() for name, meta in self.plugin_metadata.items()},
            'categories': self.list_plugins(),
            'export_timestamp': str(Path(output_path).stat().st_mtime)
        }
        
        with open(output_path, 'w') as f:
            json.dump(registry_data, f, indent=2)
            
        logger.info(f"Plugin registry exported to: {output_path}")


# Global plugin manager instance
plugin_manager = PluginManager()


# Convenience functions for plugin registration
def register_ml_detector(detector_class: Type[MLDetectorPlugin]) -> None:
    """Register a custom ML detector plugin."""
    plugin_manager.register_ml_detector(detector_class)


def register_output_format(format_class: Type[OutputFormatPlugin]) -> None:
    """Register a custom output format plugin.""" 
    plugin_manager.register_output_format(format_class)


def register_log_parser(parser_class: Type[LogParserPlugin]) -> None:
    """Register a custom log parser plugin."""
    plugin_manager.register_log_parser(parser_class)


def load_plugins(plugins_dir: str) -> None:
    """Load plugins from directory."""
    plugin_manager.load_plugins_from_directory(Path(plugins_dir))


def get_ml_detector(name: str, config: Dict = None) -> MLDetectorPlugin:
    """Get ML detector plugin instance."""
    return plugin_manager.get_ml_detector(name, config)


def get_output_format(name: str, config: Dict = None) -> OutputFormatPlugin:
    """Get output format plugin instance."""
    return plugin_manager.get_output_format(name, config)


def get_log_parser(name: str, config: Dict = None) -> LogParserPlugin:
    """Get log parser plugin instance."""
    return plugin_manager.get_log_parser(name, config)
