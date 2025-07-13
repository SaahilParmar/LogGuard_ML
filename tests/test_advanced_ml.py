"""
Tests for advanced ML module to improve coverage.

This test module focuses on testing the advanced ML functionality including
feature engineering, ensemble methods, and online detection.
"""

import pytest
import tempfile
import os
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from pathlib import Path

from logguard_ml.core.advanced_ml import (
    AdvancedAnomalyDetector, AnomalyDetectionError, 
    FeatureEngineer, EnsembleDetector
)


class TestAdvancedAnomalyDetectorExtended:
    """Extended tests for AdvancedAnomalyDetector."""
    
    def test_init_unsupported_algorithm(self):
        """Test initialization with unsupported algorithm."""
        config = {
            "ml_model": {
                "algorithm": "unsupported_algorithm",
                "contamination": 0.1
            }
        }
        
        detector = AdvancedAnomalyDetector(config)
        # Should fall back to isolation_forest
        assert detector.algorithm == "isolation_forest"
    
    def test_init_one_class_svm(self):
        """Test initialization with OneClassSVM algorithm."""
        config = {
            "ml_model": {
                "algorithm": "one_class_svm",
                "contamination": 0.1,
                "random_state": 42,
                "use_pca": False
            }
        }
        
        detector = AdvancedAnomalyDetector(config)
        assert detector.algorithm == "one_class_svm"
        assert detector.use_dimensionality_reduction is False
    
    def test_init_local_outlier_factor(self):
        """Test initialization with LocalOutlierFactor algorithm."""
        config = {
            "ml_model": {
                "algorithm": "local_outlier_factor",
                "contamination": 0.05
            }
        }
        
        detector = AdvancedAnomalyDetector(config)
        assert detector.algorithm == "local_outlier_factor"
    
    def test_init_ensemble(self):
        """Test initialization with ensemble algorithm."""
        config = {
            "ml_model": {
                "algorithm": "ensemble",
                "contamination": 0.1
            }
        }
        
        detector = AdvancedAnomalyDetector(config)
        assert detector.algorithm == "ensemble"
    
    def test_detect_anomalies_empty_dataframe(self):
        """Test anomaly detection with empty dataframe."""
        config = {"ml_model": {"algorithm": "isolation_forest"}}
        detector = AdvancedAnomalyDetector(config)
        
        df = pd.DataFrame()
        
        # Should return empty DataFrame with anomaly columns, not raise error
        result_df = detector.detect_anomalies(df)
        assert len(result_df) == 0
        assert 'is_anomaly' in result_df.columns
        assert 'anomaly_score' in result_df.columns
    
    def test_detect_anomalies_invalid_columns(self):
        """Test anomaly detection with missing required columns."""
        config = {"ml_model": {"algorithm": "isolation_forest"}}
        detector = AdvancedAnomalyDetector(config)
        
        df = pd.DataFrame({'wrong_column': [1, 2, 3]})
        
        # Should handle gracefully and return result with anomaly columns
        result_df = detector.detect_anomalies(df)
        assert len(result_df) == 3
        assert 'is_anomaly' in result_df.columns
    
    def test_detect_anomalies_minimal_data(self):
        """Test anomaly detection with minimal valid data."""
        config = {"ml_model": {"algorithm": "isolation_forest", "contamination": 0.1}}
        detector = AdvancedAnomalyDetector(config)
        
        df = pd.DataFrame({
            'timestamp': ['2024-01-01 12:00:00', '2024-01-01 12:01:00'],
            'level': ['INFO', 'ERROR'], 
            'message': ['normal message', 'error message'],
            'line_number': [1, 2]
        })
        
        result_df = detector.detect_anomalies(df)
        
        assert len(result_df) == 2
        assert 'is_anomaly' in result_df.columns
        assert 'anomaly_score' in result_df.columns
    
    def test_feature_engineering_text_features(self):
        """Test that feature engineering works with text data."""
        config = {
            "ml_model": {
                "algorithm": "isolation_forest",
                "max_features": 100
            }
        }
        detector = AdvancedAnomalyDetector(config)
        
        # Create data with sufficient volume for text processing
        df = pd.DataFrame({
            'timestamp': ['2024-01-01 12:00:00'] * 10,
            'level': ['INFO', 'ERROR', 'WARN'] * 3 + ['INFO'],
            'message': [
                'normal operation completed successfully',
                'database connection failed with timeout',
                'high memory usage detected in system',
                'user authentication successful',
                'file not found error occurred',
                'network connection established',
                'cache miss detected',
                'backup process started',
                'service restart required',
                'configuration updated'
            ],
            'line_number': list(range(1, 11))
        })
        
        # Just test that detection completes without error
        result_df = detector.detect_anomalies(df)
        assert len(result_df) == 10
        assert 'is_anomaly' in result_df.columns
    
    def test_save_load_model(self):
        """Test saving and loading model."""
        config = {"ml_model": {"algorithm": "isolation_forest", "contamination": 0.1}}
        detector = AdvancedAnomalyDetector(config)
        
        # Create and fit model with sample data
        df = pd.DataFrame({
            'timestamp': ['2024-01-01 12:00:00'] * 10,
            'level': ['INFO'] * 8 + ['ERROR'] * 2,
            'message': [f'message {i}' for i in range(10)],
            'line_number': list(range(1, 11))
        })
        
        # Fit the model
        result_df = detector.detect_anomalies(df)
        assert detector.is_fitted
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name
        
        try:
            detector.save_model(temp_path)
            assert Path(temp_path).exists()
            
            # Load model into new detector
            new_detector = AdvancedAnomalyDetector(config)
            new_detector.load_model(temp_path)
            
            assert new_detector.is_fitted
        finally:
            os.unlink(temp_path)
    
    def test_save_model_unfitted(self):
        """Test saving unfitted model raises error."""
        config = {"ml_model": {"algorithm": "isolation_forest"}}
        detector = AdvancedAnomalyDetector(config)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl') as f:
            with pytest.raises(ValueError, match="Model must be fitted before saving"):
                detector.save_model(f.name)
    
    def test_dimensionality_reduction(self):
        """Test dimensionality reduction."""
        config = {
            "ml_model": {
                "algorithm": "isolation_forest",
                "use_pca": True,
                "n_components": 5,
                "contamination": 0.1
            }
        }
        detector = AdvancedAnomalyDetector(config)
        
        # Create high-dimensional data
        df = pd.DataFrame({
            'timestamp': ['2024-01-01 12:00:00'] * 20,
            'level': ['INFO'] * 15 + ['ERROR'] * 5,
            'message': [f'message with lots of text content {i}' for i in range(20)],
            'line_number': list(range(1, 21))
        })
        
        result_df = detector.detect_anomalies(df)
        
        assert len(result_df) == 20
        assert 'is_anomaly' in result_df.columns


class TestFeatureEngineer:
    """Test FeatureEngineer functionality."""
    
    def test_init(self):
        """Test FeatureEngineer initialization."""
        config = {
            "ml_model": {
                "max_features": 1000,
                "use_hashing_vectorizer": True,
                "ngram_range": [1, 3]
            }
        }
        
        engineer = FeatureEngineer(config)
        assert engineer.config == config
        assert engineer.max_features == 1000
        assert engineer.use_hashing is True
        assert engineer.ngram_range == (1, 3)
    
    def test_init_defaults(self):
        """Test FeatureEngineer with default configuration."""
        config = {}
        engineer = FeatureEngineer(config)
        
        # Should use defaults
        assert engineer.max_features == 5000
        assert engineer.use_hashing is False
        assert engineer.ngram_range == (1, 2)
    
    def test_extract_features(self):
        """Test feature extraction from DataFrame."""
        config = {"ml_model": {"max_features": 100}}
        engineer = FeatureEngineer(config)
        
        df = pd.DataFrame({
            'timestamp': ['2024-01-01 09:30:00', '2024-01-01 14:45:00', '2024-01-01 22:15:00'],
            'level': ['INFO', 'ERROR', 'WARN'],
            'message': [
                'normal operation completed successfully',
                'database connection timeout error',
                'memory usage is getting high'
            ],
            'line_number': [1, 2, 3]
        })
        
        features, feature_names = engineer.extract_features(df)
        assert features.shape[0] == 3
        assert features.shape[1] > 0
        assert len(feature_names) > 0
    
    def test_extract_features_empty_df(self):
        """Test feature extraction with empty DataFrame."""
        config = {}
        engineer = FeatureEngineer(config)
        
        df = pd.DataFrame()
        features, feature_names = engineer.extract_features(df)
        
        assert features.shape == (0, 0)
        assert len(feature_names) == 0


class TestEnsembleDetector:
    """Test EnsembleDetector functionality."""
    
    def test_init(self):
        """Test EnsembleDetector initialization."""
        config = {
            "ml_model": {
                "contamination": 0.1,
                "random_state": 42
            }
        }
        
        detector = EnsembleDetector(config)
        assert len(detector.detectors) == 3  # isolation_forest, one_class_svm, local_outlier_factor
        assert detector.contamination == 0.1
        assert detector.random_state == 42
        assert detector.is_fitted is False
    
    def test_init_defaults(self):
        """Test EnsembleDetector with default configuration."""
        config = {}
        detector = EnsembleDetector(config)
        
        # Should use default contamination
        assert detector.contamination == 0.05
        assert detector.random_state == 42
        assert len(detector.detectors) >= 2
    
    def test_fit_predict(self):
        """Test ensemble fit and predict."""
        config = {
            "ml_model": {
                "contamination": 0.1
            }
        }
        
        detector = EnsembleDetector(config)
        
        # Create test data
        np.random.seed(42)
        features = np.random.rand(50, 5)
        
        # Fit the ensemble
        detector.fit(features)
        assert detector.is_fitted is True
        
        # Predict on same data
        predictions = detector.predict(features)
        assert len(predictions) == 50
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_predict_unfitted(self):
        """Test predict on unfitted ensemble."""
        config = {}
        detector = EnsembleDetector(config)
        
        features = np.random.rand(10, 3)
        
        with pytest.raises(ValueError, match="Ensemble must be fitted before prediction"):
            detector.predict(features)
    
    def test_score_samples(self):
        """Test ensemble scoring."""
        config = {"ml_model": {"contamination": 0.1}}
        detector = EnsembleDetector(config)
        
        # Create and fit on test data
        np.random.seed(42)
        features = np.random.rand(30, 4)
        detector.fit(features)
        
        # Get scores
        scores = detector.score_samples(features)
        assert len(scores) == 30
        assert all(isinstance(score, (int, float)) for score in scores)
    
    def test_score_samples_unfitted(self):
        """Test scoring on unfitted ensemble."""
        config = {}
        detector = EnsembleDetector(config)
        
        features = np.random.rand(10, 3)
        
        with pytest.raises(ValueError, match="Ensemble must be fitted before scoring"):
            detector.score_samples(features)
