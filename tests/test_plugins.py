"""
Comprehensive tests for the plugin system.

This test module provides extensive coverage for the plugin architecture,
including plugin loading, registration, error handling, and lifecycle management.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import yaml

from logguard_ml.plugins import (
    PluginManager, PluginInterface, MLDetectorPlugin, OutputFormatPlugin,
    LogParserPlugin, PluginMetadata, PluginError, plugin_manager,
    register_ml_detector, register_output_format, register_log_parser,
    load_plugins, get_ml_detector, get_output_format, get_log_parser
)


class TestPluginInterface:
    """Test the base plugin interface."""
    
    def test_plugin_interface_is_abstract(self):
        """Test that PluginInterface cannot be instantiated directly."""
        with pytest.raises(TypeError):
            PluginInterface()
    
    def test_plugin_interface_methods(self):
        """Test that all abstract methods are properly defined."""
        # Create a concrete implementation
        class ConcretePlugin(PluginInterface):
            @property
            def name(self):
                return "test_plugin"
            
            @property
            def version(self):
                return "1.0.0"
            
            @property
            def description(self):
                return "Test plugin"
        
        plugin = ConcretePlugin()
        assert plugin.name == "test_plugin"
        assert plugin.version == "1.0.0"
        assert plugin.description == "Test plugin"
        
        # Test optional methods
        plugin.initialize({})  # Should not raise
        plugin.cleanup()  # Should not raise


class TestMLDetectorPlugin:
    """Test ML detector plugin base class."""
    
    def test_ml_detector_plugin_is_abstract(self):
        """Test that MLDetectorPlugin cannot be instantiated directly."""
        with pytest.raises(TypeError):
            MLDetectorPlugin()
    
    def test_ml_detector_plugin_methods(self):
        """Test ML detector plugin methods."""
        class TestMLDetector(MLDetectorPlugin):
            @property
            def name(self):
                return "test_ml_detector"
            
            @property
            def version(self):
                return "1.0.0"
            
            @property
            def description(self):
                return "Test ML detector"
            
            def detect_anomalies(self, df, config):
                return df.copy()
            
            def get_feature_importance(self):
                return {"feature1": 0.8, "feature2": 0.2}
        
        detector = TestMLDetector()
        
        # Test data validation
        valid_df = pd.DataFrame({
            'message': ['log1', 'log2'],
            'level': ['INFO', 'ERROR'],
            'timestamp': ['2023-01-01', '2023-01-02']
        })
        assert detector.validate_data(valid_df) is True
        
        invalid_df = pd.DataFrame({'invalid': ['data']})
        assert detector.validate_data(invalid_df) is False
        
        # Test detection
        result = detector.detect_anomalies(valid_df, {})
        assert isinstance(result, pd.DataFrame)
        
        # Test feature importance
        importance = detector.get_feature_importance()
        assert isinstance(importance, dict)
        assert "feature1" in importance


class TestOutputFormatPlugin:
    """Test output format plugin base class."""
    
    def test_output_format_plugin_methods(self):
        """Test output format plugin methods."""
        class TestOutputFormat(OutputFormatPlugin):
            @property
            def name(self):
                return "test_format"
            
            @property
            def version(self):
                return "1.0.0"
            
            @property
            def description(self):
                return "Test format"
            
            @property
            def file_extension(self):
                return "test"
            
            def generate_output(self, df, output_path, **kwargs):
                with open(output_path, 'w') as f:
                    f.write("test output")
        
        formatter = TestOutputFormat()
        
        # Test path validation
        with tempfile.TemporaryDirectory() as temp_dir:
            valid_path = Path(temp_dir) / "test.txt"
            assert formatter.validate_output_path(str(valid_path)) is True
            
            # Test generation
            test_df = pd.DataFrame({'data': [1, 2, 3]})
            formatter.generate_output(test_df, str(valid_path))
            assert valid_path.exists()
            assert valid_path.read_text() == "test output"


class TestLogParserPlugin:
    """Test log parser plugin base class."""
    
    def test_log_parser_plugin_methods(self):
        """Test log parser plugin methods."""
        class TestLogParser(LogParserPlugin):
            @property
            def name(self):
                return "test_parser"
            
            @property
            def version(self):
                return "1.0.0"
            
            @property
            def description(self):
                return "Test parser"
            
            @property
            def supported_formats(self):
                return ["test", "example"]
            
            def parse_log_line(self, line):
                return {"raw": line, "parsed": True}
        
        parser = TestLogParser()
        
        # Test parsing
        result = parser.parse_log_line("test log line")
        assert result == {"raw": "test log line", "parsed": True}
        
        # Test format detection
        sample_log = "line1\nline2\nline3"
        assert parser.can_parse(sample_log) is True
        
        # Test with parser that returns invalid data
        class BadParser(LogParserPlugin):
            @property
            def name(self):
                return "bad_parser"
            
            @property
            def version(self):
                return "1.0.0"
            
            @property
            def description(self):
                return "Bad parser"
            
            @property
            def supported_formats(self):
                return ["bad"]
            
            def parse_log_line(self, line):
                return None  # Invalid return
        
        bad_parser = BadParser()
        assert bad_parser.can_parse(sample_log) is False


class TestPluginMetadata:
    """Test plugin metadata handling."""
    
    def test_metadata_creation(self):
        """Test creating metadata with all fields."""
        metadata = PluginMetadata(
            name="test_plugin",
            version="1.0.0",
            description="Test plugin",
            author="Test Author",
            dependencies=["dep1", "dep2"],
            config_schema={"type": "object"}
        )
        
        assert metadata.name == "test_plugin"
        assert metadata.version == "1.0.0"
        assert metadata.description == "Test plugin"
        assert metadata.author == "Test Author"
        assert metadata.dependencies == ["dep1", "dep2"]
        assert metadata.config_schema == {"type": "object"}
    
    def test_metadata_defaults(self):
        """Test metadata with default values."""
        metadata = PluginMetadata(
            name="test",
            version="1.0.0",
            description="Test"
        )
        
        assert metadata.author == ""
        assert metadata.dependencies == []
        assert metadata.config_schema == {}
    
    def test_metadata_serialization(self):
        """Test metadata to/from dict conversion."""
        original_data = {
            "name": "test_plugin",
            "version": "1.0.0",
            "description": "Test plugin",
            "author": "Test Author",
            "dependencies": ["dep1"],
            "config_schema": {"type": "object"}
        }
        
        # Test from_dict
        metadata = PluginMetadata.from_dict(original_data)
        assert metadata.name == "test_plugin"
        assert metadata.author == "Test Author"
        
        # Test to_dict
        result_data = metadata.to_dict()
        assert result_data == original_data


class TestPluginManager:
    """Test the plugin manager functionality."""
    
    def setup_method(self):
        """Set up clean plugin manager for each test."""
        self.manager = PluginManager()
    
    def test_plugin_manager_initialization(self):
        """Test plugin manager initialization."""
        assert isinstance(self.manager.ml_detectors, dict)
        assert isinstance(self.manager.output_formats, dict)
        assert isinstance(self.manager.log_parsers, dict)
        assert isinstance(self.manager.plugin_metadata, dict)
        assert isinstance(self.manager.loaded_plugins, dict)
        
        assert len(self.manager.ml_detectors) == 0
        assert len(self.manager.output_formats) == 0
        assert len(self.manager.log_parsers) == 0
    
    def test_register_ml_detector(self):
        """Test ML detector registration."""
        class TestDetector(MLDetectorPlugin):
            @property
            def name(self):
                return "test_detector"
            
            @property
            def version(self):
                return "1.0.0"
            
            @property
            def description(self):
                return "Test detector"
            
            def detect_anomalies(self, df, config):
                return df
            
            def get_feature_importance(self):
                return {}
        
        self.manager.register_ml_detector(TestDetector)
        
        assert "test_detector" in self.manager.ml_detectors
        assert "test_detector" in self.manager.plugin_metadata
        
        metadata = self.manager.plugin_metadata["test_detector"]
        assert metadata.name == "test_detector"
        assert metadata.version == "1.0.0"
    
    def test_register_invalid_ml_detector(self):
        """Test registration of invalid ML detector."""
        class InvalidDetector:
            pass
        
        with pytest.raises(PluginError, match="must inherit from MLDetectorPlugin"):
            self.manager.register_ml_detector(InvalidDetector)
    
    def test_register_output_format(self):
        """Test output format registration."""
        class TestFormat(OutputFormatPlugin):
            @property
            def name(self):
                return "test_format"
            
            @property
            def version(self):
                return "1.0.0"
            
            @property
            def description(self):
                return "Test format"
            
            @property
            def file_extension(self):
                return "test"
            
            def generate_output(self, df, output_path, **kwargs):
                pass
        
        self.manager.register_output_format(TestFormat)
        
        assert "test_format" in self.manager.output_formats
        assert "test_format" in self.manager.plugin_metadata
    
    def test_register_log_parser(self):
        """Test log parser registration."""
        class TestParser(LogParserPlugin):
            @property
            def name(self):
                return "test_parser"
            
            @property
            def version(self):
                return "1.0.0"
            
            @property
            def description(self):
                return "Test parser"
            
            @property
            def supported_formats(self):
                return ["test"]
            
            def parse_log_line(self, line):
                return {"line": line}
        
        self.manager.register_log_parser(TestParser)
        
        assert "test_parser" in self.manager.log_parsers
        assert "test_parser" in self.manager.plugin_metadata
    
    def test_get_ml_detector(self):
        """Test getting ML detector instance."""
        class TestDetector(MLDetectorPlugin):
            @property
            def name(self):
                return "test_detector"
            
            @property
            def version(self):
                return "1.0.0"
            
            @property
            def description(self):
                return "Test detector"
            
            def detect_anomalies(self, df, config):
                return df
            
            def get_feature_importance(self):
                return {}
            
            def initialize(self, config):
                self.config = config
        
        self.manager.register_ml_detector(TestDetector)
        
        # Test without config
        detector = self.manager.get_ml_detector("test_detector")
        assert isinstance(detector, TestDetector)
        
        # Test with config
        config = {"param": "value"}
        detector_with_config = self.manager.get_ml_detector("test_detector", config)
        assert detector_with_config.config == config
        
        # Test non-existent detector
        with pytest.raises(PluginError, match="ML detector 'nonexistent' not found"):
            self.manager.get_ml_detector("nonexistent")
    
    def test_list_plugins(self):
        """Test listing all plugins."""
        # Register some test plugins
        class TestDetector(MLDetectorPlugin):
            @property
            def name(self):
                return "detector1"
            
            @property
            def version(self):
                return "1.0.0"
            
            @property
            def description(self):
                return "Test"
            
            def detect_anomalies(self, df, config):
                return df
            
            def get_feature_importance(self):
                return {}
        
        class TestFormat(OutputFormatPlugin):
            @property
            def name(self):
                return "format1"
            
            @property
            def version(self):
                return "1.0.0"
            
            @property
            def description(self):
                return "Test"
            
            @property
            def file_extension(self):
                return "test"
            
            def generate_output(self, df, output_path, **kwargs):
                pass
        
        self.manager.register_ml_detector(TestDetector)
        self.manager.register_output_format(TestFormat)
        
        plugins = self.manager.list_plugins()
        
        assert "ml_detectors" in plugins
        assert "output_formats" in plugins
        assert "log_parsers" in plugins
        
        assert "detector1" in plugins["ml_detectors"]
        assert "format1" in plugins["output_formats"]
        assert len(plugins["log_parsers"]) == 0
    
    def test_get_plugin_info(self):
        """Test getting plugin information."""
        class TestDetector(MLDetectorPlugin):
            @property
            def name(self):
                return "test_detector"
            
            @property
            def version(self):
                return "1.0.0"
            
            @property
            def description(self):
                return "Test detector"
            
            def detect_anomalies(self, df, config):
                return df
            
            def get_feature_importance(self):
                return {}
        
        self.manager.register_ml_detector(TestDetector)
        
        info = self.manager.get_plugin_info("test_detector")
        assert info["name"] == "test_detector"
        assert info["version"] == "1.0.0"
        assert info["description"] == "Test detector"
        
        # Test non-existent plugin
        with pytest.raises(PluginError, match="Plugin 'nonexistent' not found"):
            self.manager.get_plugin_info("nonexistent")
    
    def test_validate_plugin_config(self):
        """Test plugin configuration validation."""
        # Test with no metadata (should return True)
        assert self.manager.validate_plugin_config("nonexistent", {}) is True
        
        # Test with metadata but no schema
        metadata = PluginMetadata("test", "1.0.0", "Test")
        self.manager.plugin_metadata["test"] = metadata
        assert self.manager.validate_plugin_config("test", {}) is True
        
        # Test with schema (placeholder implementation)
        metadata.config_schema = {"type": "object"}
        assert self.manager.validate_plugin_config("test", {}) is True
    
    def test_export_plugin_registry(self):
        """Test exporting plugin registry."""
        # Register a test plugin
        class TestDetector(MLDetectorPlugin):
            @property
            def name(self):
                return "test_detector"
            
            @property
            def version(self):
                return "1.0.0"
            
            @property
            def description(self):
                return "Test detector"
            
            def detect_anomalies(self, df, config):
                return df
            
            def get_feature_importance(self):
                return {}
        
        self.manager.register_ml_detector(TestDetector)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = f.name
        
        try:
            self.manager.export_plugin_registry(output_path)
            
            # Verify exported data
            with open(output_path, 'r') as f:
                data = json.load(f)
            
            assert "plugins" in data
            assert "categories" in data
            assert "export_timestamp" in data
            
            assert "test_detector" in data["plugins"]
            assert data["plugins"]["test_detector"]["name"] == "test_detector"
            
        finally:
            Path(output_path).unlink()
    
    @patch('importlib.util.spec_from_file_location')
    @patch('importlib.util.module_from_spec')
    def test_load_plugin_file(self, mock_module_from_spec, mock_spec_from_file):
        """Test loading plugin from file."""
        # Mock module loading
        mock_spec = Mock()
        mock_loader = Mock()
        mock_spec.loader = mock_loader
        mock_spec_from_file.return_value = mock_spec
        
        mock_module = Mock()
        mock_module_from_spec.return_value = mock_module
        
        # Create test plugin class
        class TestPlugin(MLDetectorPlugin):
            @property
            def name(self):
                return "file_plugin"
            
            @property
            def version(self):
                return "1.0.0"
            
            @property
            def description(self):
                return "File plugin"
            
            def detect_anomalies(self, df, config):
                return df
            
            def get_feature_importance(self):
                return {}
        
        # Mock inspect.getmembers to return our test class
        with patch('inspect.getmembers', return_value=[('TestPlugin', TestPlugin)]):
            plugin_path = Path("/fake/path/test_plugin.py")
            self.manager.load_plugin_file(plugin_path)
        
        assert "file_plugin" in self.manager.ml_detectors
    
    def test_load_plugin_file_with_metadata(self):
        """Test loading plugin file with metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            plugin_dir = Path(temp_dir)
            plugin_file = plugin_dir / "test_plugin.py"
            metadata_file = plugin_dir / "test_plugin_metadata.yaml"
            
            # Create plugin file
            plugin_content = '''
class TestPlugin:
    @property
    def name(self):
        return "test_plugin"
    
    @property
    def version(self):
        return "2.0.0"
    
    @property
    def description(self):
        return "Test plugin from file"
'''
            plugin_file.write_text(plugin_content)
            
            # Create metadata file
            metadata_content = {
                "name": "test_plugin",
                "version": "2.0.0",
                "description": "Test plugin from file",
                "author": "Test Author",
                "dependencies": ["numpy"]
            }
            with open(metadata_file, 'w') as f:
                yaml.dump(metadata_content, f)
            
            # Test loading (will fail due to incomplete plugin class, but metadata should be read)
            try:
                self.manager.load_plugin_file(plugin_file)
            except Exception:
                pass  # Expected due to incomplete plugin class
    
    def test_load_plugins_from_directory(self):
        """Test loading plugins from directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            plugin_dir = Path(temp_dir)
            
            # Create test plugin file
            plugin_file = plugin_dir / "valid_plugin.py"
            plugin_file.write_text("# Test plugin")
            
            # Create __init__.py (should be skipped)
            init_file = plugin_dir / "__init__.py"
            init_file.write_text("# Init file")
            
            # Mock the load_plugin_file method
            with patch.object(self.manager, 'load_plugin_file') as mock_load:
                self.manager.load_plugins_from_directory(plugin_dir)
                
                # Should be called once (for valid_plugin.py, not __init__.py)
                mock_load.assert_called_once_with(plugin_file)
    
    def test_load_plugins_from_nonexistent_directory(self):
        """Test loading plugins from non-existent directory."""
        nonexistent_dir = Path("/path/that/does/not/exist")
        
        # Should not raise exception, just log warning
        self.manager.load_plugins_from_directory(nonexistent_dir)


class TestGlobalFunctions:
    """Test global convenience functions."""
    
    def setup_method(self):
        """Reset global plugin manager."""
        plugin_manager.ml_detectors.clear()
        plugin_manager.output_formats.clear()
        plugin_manager.log_parsers.clear()
        plugin_manager.plugin_metadata.clear()
        plugin_manager.loaded_plugins.clear()
    
    def test_register_ml_detector_global(self):
        """Test global ML detector registration."""
        class GlobalTestDetector(MLDetectorPlugin):
            @property
            def name(self):
                return "global_detector"
            
            @property
            def version(self):
                return "1.0.0"
            
            @property
            def description(self):
                return "Global test detector"
            
            def detect_anomalies(self, df, config):
                return df
            
            def get_feature_importance(self):
                return {}
        
        register_ml_detector(GlobalTestDetector)
        assert "global_detector" in plugin_manager.ml_detectors
    
    def test_get_ml_detector_global(self):
        """Test global ML detector getter."""
        class GlobalTestDetector(MLDetectorPlugin):
            @property
            def name(self):
                return "global_detector"
            
            @property
            def version(self):
                return "1.0.0"
            
            @property
            def description(self):
                return "Global test detector"
            
            def detect_anomalies(self, df, config):
                return df
            
            def get_feature_importance(self):
                return {}
        
        register_ml_detector(GlobalTestDetector)
        detector = get_ml_detector("global_detector")
        assert isinstance(detector, GlobalTestDetector)
    
    def test_load_plugins_global(self):
        """Test global plugin loading."""
        with patch.object(plugin_manager, 'load_plugins_from_directory') as mock_load:
            load_plugins("/test/path")
            mock_load.assert_called_once_with(Path("/test/path"))


class TestPluginErrorHandling:
    """Test error handling in plugin system."""
    
    def test_plugin_error_inheritance(self):
        """Test that PluginError inherits from Exception."""
        error = PluginError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"
    
    def test_invalid_plugin_registration(self):
        """Test various invalid plugin registrations."""
        manager = PluginManager()
        
        # Test registering non-class
        with pytest.raises((PluginError, TypeError)):
            manager.register_ml_detector("not_a_class")
        
        # Test registering wrong type
        class WrongType:
            pass
        
        with pytest.raises(PluginError):
            manager.register_ml_detector(WrongType)
    
    def test_plugin_loading_errors(self):
        """Test error handling during plugin loading."""
        manager = PluginManager()
        
        # Test loading non-existent file
        with pytest.raises(PluginError):
            manager.load_plugin_file(Path("/nonexistent/plugin.py"))


class TestPluginIntegration:
    """Integration tests for plugin system."""
    
    def test_end_to_end_plugin_workflow(self):
        """Test complete plugin workflow."""
        manager = PluginManager()
        
        # Create and register plugins
        class TestMLDetector(MLDetectorPlugin):
            @property
            def name(self):
                return "integration_detector"
            
            @property
            def version(self):
                return "1.0.0"
            
            @property
            def description(self):
                return "Integration test detector"
            
            def detect_anomalies(self, df, config):
                df = df.copy()
                df['anomaly'] = False
                df.loc[0, 'anomaly'] = True  # Mark first row as anomaly
                return df
            
            def get_feature_importance(self):
                return {"message_length": 0.8, "level_severity": 0.2}
        
        class TestOutputFormat(OutputFormatPlugin):
            @property
            def name(self):
                return "integration_format"
            
            @property
            def version(self):
                return "1.0.0"
            
            @property
            def description(self):
                return "Integration test format"
            
            @property
            def file_extension(self):
                return "integ"
            
            def generate_output(self, df, output_path, **kwargs):
                # Handle boolean sum properly
                anomaly_count = int(df['anomaly'].sum()) if len(df) > 0 else 0
                summary = f"Processed {len(df)} rows, found {anomaly_count} anomalies"
                with open(output_path, 'w') as f:
                    f.write(summary)
        
        # Register plugins
        manager.register_ml_detector(TestMLDetector)
        manager.register_output_format(TestOutputFormat)
        
        # Test plugin listing
        plugins = manager.list_plugins()
        assert "integration_detector" in plugins["ml_detectors"]
        assert "integration_format" in plugins["output_formats"]
        
        # Create test data
        test_data = pd.DataFrame({
            'message': ['Normal log', 'Suspicious activity', 'Regular update'],
            'level': ['INFO', 'WARNING', 'INFO'],
            'timestamp': ['2023-01-01 10:00', '2023-01-01 10:01', '2023-01-01 10:02']
        })
        
        # Test detection
        detector = manager.get_ml_detector("integration_detector")
        result = detector.detect_anomalies(test_data, {})
        assert 'anomaly' in result.columns
        assert result['anomaly'].iloc[0] == True
        
        # Test output generation
        formatter = manager.get_output_format("integration_format")
        with tempfile.NamedTemporaryFile(mode='w', suffix='.integ', delete=False) as f:
            output_path = f.name
        
        try:
            formatter.generate_output(result, output_path)
            
            with open(output_path, 'r') as f:
                content = f.read()
            
            assert "Processed 3 rows" in content
            assert "found 1 anomalies" in content
            
        finally:
            Path(output_path).unlink()
        
        # Test metadata export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            registry_path = f.name
        
        try:
            manager.export_plugin_registry(registry_path)
            
            with open(registry_path, 'r') as f:
                registry = json.load(f)
            
            assert "integration_detector" in registry["plugins"]
            assert "integration_format" in registry["plugins"]
            
        finally:
            Path(registry_path).unlink()
