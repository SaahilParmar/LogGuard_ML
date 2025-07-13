"""
Advanced Machine Learning Models for LogGuard ML

This module provides enhanced ML algorithms with improved performance,
accuracy, and scalability for log anomaly detection.

Classes:
    AdvancedAnomalyDetector: Enhanced anomaly detection with multiple algorithms
    EnsembleDetector: Ensemble methods for improved accuracy
    OnlineDetector: Streaming anomaly detection for real-time processing
    FeatureEngineer: Advanced feature extraction and engineering

Example:
    >>> from logguard_ml.core.advanced_ml import AdvancedAnomalyDetector
    >>> detector = AdvancedAnomalyDetector(config)
    >>> anomalies = detector.detect_anomalies(df)
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from scipy import stats
import hashlib

from .performance import PerformanceMonitor, profile_function, MemoryProfiler

logger = logging.getLogger(__name__)

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


class AnomalyDetectionError(Exception):
    """Custom exception for anomaly detection errors."""
    pass


class AdvancedAnomalyDetector:
    """
    Advanced anomaly detector with multiple algorithms and optimization.
    
    Supports multiple detection algorithms, ensemble methods, and advanced
    feature engineering for improved anomaly detection accuracy.
    """
    
    SUPPORTED_ALGORITHMS = {
        'isolation_forest': IsolationForest,
        'one_class_svm': OneClassSVM,
        'local_outlier_factor': LocalOutlierFactor,
        'ensemble': 'EnsembleDetector'
    }
    
    def __init__(self, config: Dict):
        """
        Initialize advanced anomaly detector.
        
        Args:
            config: Configuration dictionary with ML parameters
        """
        self.config = config
        self.ml_config = config.get("ml_model", {})
        
        # Algorithm selection
        self.algorithm = self.ml_config.get("algorithm", "isolation_forest")
        if self.algorithm not in self.SUPPORTED_ALGORITHMS:
            logger.warning(f"Unsupported algorithm '{self.algorithm}', using 'isolation_forest'")
            self.algorithm = "isolation_forest"
        
        # Model parameters
        self.contamination = self.ml_config.get("contamination", 0.05)
        self.random_state = self.ml_config.get("random_state", 42)
        
        # Initialize components
        self.model = None
        self.scaler = None
        self.feature_engineer = FeatureEngineer(config)
        self.is_fitted = False
        
        # Performance optimization
        self.use_dimensionality_reduction = self.ml_config.get("use_pca", True)
        self.max_features = self.ml_config.get("max_features", 10000)
        self.n_components = self.ml_config.get("n_components", 50)
        
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the ML model based on configuration."""
        if self.algorithm == "isolation_forest":
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
                n_estimators=self.ml_config.get("n_estimators", 100),
                max_samples=self.ml_config.get("max_samples", "auto"),
                n_jobs=-1  # Use all available cores
            )
            self.scaler = RobustScaler()
            
        elif self.algorithm == "one_class_svm":
            self.model = OneClassSVM(
                nu=self.contamination,
                kernel=self.ml_config.get("kernel", "rbf"),
                gamma=self.ml_config.get("gamma", "scale")
            )
            self.scaler = StandardScaler()
            
        elif self.algorithm == "local_outlier_factor":
            self.model = LocalOutlierFactor(
                contamination=self.contamination,
                n_neighbors=self.ml_config.get("n_neighbors", 20),
                novelty=True,  # For prediction on new data
                n_jobs=-1
            )
            self.scaler = StandardScaler()
            
        elif self.algorithm == "ensemble":
            self.model = EnsembleDetector(self.config)
            self.scaler = RobustScaler()
            
        logger.info(f"Initialized {self.algorithm} detector")
    
    @profile_function
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies in log data with advanced feature engineering.
        
        Args:
            df: Input DataFrame with log data
            
        Returns:
            DataFrame with anomaly detection results
        """
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return self._add_empty_anomaly_columns(df.copy())
        
        logger.info(f"Starting anomaly detection on {len(df)} log entries using {self.algorithm}")
        
        with PerformanceMonitor() as monitor:
            try:
                # Feature engineering
                logger.debug("Extracting features...")
                features, feature_names = self.feature_engineer.extract_features(df)
                
                if features.size == 0:
                    logger.warning("No features could be extracted")
                    return self._add_empty_anomaly_columns(df.copy())
                
                # Memory optimization
                features = MemoryProfiler.optimize_dataframe(
                    pd.DataFrame(features, columns=feature_names)
                ).values
                
                # Dimensionality reduction if needed
                if self.use_dimensionality_reduction and features.shape[1] > self.n_components:
                    features = self._reduce_dimensionality(features)
                
                # Scale features
                features_scaled = self.scaler.fit_transform(features)
                
                # Train and predict
                logger.debug("Training model and making predictions...")
                predictions, scores = self._fit_predict(features_scaled)
                
                # Create result DataFrame
                result_df = df.copy()
                result_df["is_anomaly"] = predictions
                result_df["anomaly_score"] = scores
                
                # Add confidence scores
                result_df["confidence"] = self._calculate_confidence(scores)
                
                # Add feature importances if available
                if hasattr(self.model, 'feature_importances_'):
                    self._add_feature_importance_info(result_df, feature_names)
                
                self.is_fitted = True
                
                # Log results
                anomaly_count = int(predictions.sum())
                anomaly_percentage = (anomaly_count / len(df)) * 100
                
                logger.info(f"Anomaly detection completed:")
                logger.info(f"  - Found {anomaly_count} anomalies ({anomaly_percentage:.2f}%)")
                logger.info(f"  - Used {features.shape[1]} features")
                logger.info(f"  - Processing time: {monitor.get_execution_time():.2f}s")
                
                return result_df
                
            except Exception as e:
                logger.error(f"Anomaly detection failed: {e}")
                raise
    
    def _reduce_dimensionality(self, features: np.ndarray) -> np.ndarray:
        """Reduce feature dimensionality using PCA or SVD."""
        logger.debug(f"Reducing dimensionality from {features.shape[1]} to {self.n_components}")
        
        # Use TruncatedSVD for sparse matrices, PCA for dense
        if hasattr(features, 'sparse') or np.isnan(features).sum() > 0:
            reducer = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
        else:
            reducer = PCA(n_components=self.n_components, random_state=self.random_state)
        
        reduced_features = reducer.fit_transform(features)
        
        # Log explained variance if available
        if hasattr(reducer, 'explained_variance_ratio_'):
            explained_variance = reducer.explained_variance_ratio_.sum()
            logger.debug(f"Dimensionality reduction: retained {explained_variance:.2%} of variance")
        
        return reduced_features
    
    def _fit_predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit model and make predictions."""
        if self.algorithm == "local_outlier_factor":
            # LOF requires special handling
            self.model.fit(features)
            predictions = self.model.predict(features)
            scores = self.model.negative_outlier_factor_
            
            # Convert LOF output to binary predictions
            predictions = (predictions == -1).astype(int)
            
        else:
            # Standard fit/predict pattern
            self.model.fit(features)
            predictions = self.model.predict(features)
            
            # Get anomaly scores
            if hasattr(self.model, 'score_samples'):
                scores = self.model.score_samples(features)
            elif hasattr(self.model, 'decision_function'):
                scores = self.model.decision_function(features)
            else:
                scores = np.zeros(len(features))
            
            # Convert predictions to binary format
            if self.algorithm == "isolation_forest":
                predictions = (predictions == -1).astype(int)
            else:
                predictions = (predictions <= 0).astype(int)
        
        return predictions, scores
    
    def _calculate_confidence(self, scores: np.ndarray) -> np.ndarray:
        """Calculate confidence scores from anomaly scores."""
        if len(scores) == 0:
            return np.array([])
        
        # Normalize scores to [0, 1] range
        scores_normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        # Convert to confidence (higher score = higher confidence)
        confidence = 1 - scores_normalized
        
        return confidence
    
    def _add_feature_importance_info(self, df: pd.DataFrame, feature_names: List[str]):
        """Add feature importance information to results."""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            top_features = sorted(
                zip(feature_names, importances),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            logger.debug("Top 5 important features:")
            for name, importance in top_features:
                logger.debug(f"  {name}: {importance:.4f}")
    
    def _add_empty_anomaly_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add empty anomaly columns to DataFrame."""
        df["is_anomaly"] = 0
        df["anomaly_score"] = 0.0
        df["confidence"] = 1.0
        return df
    
    def save_model(self, filepath: Union[str, Path]):
        """Save trained model to disk."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_engineer': self.feature_engineer,
            'config': self.config,
            'algorithm': self.algorithm,
            'is_fitted': self.is_fitted
        };
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: Union[str, Path]):
        """Load trained model from disk."""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_engineer = model_data['feature_engineer']
        self.config = model_data['config']
        self.algorithm = model_data['algorithm']
        self.is_fitted = model_data['is_fitted']
        
        logger.info(f"Model loaded from {filepath}")


class EnsembleDetector:
    """Ensemble anomaly detector combining multiple algorithms."""
    
    def __init__(self, config: Dict):
        """Initialize ensemble detector."""
        self.config = config
        self.ml_config = config.get("ml_model", {})
        self.contamination = self.ml_config.get("contamination", 0.05)
        self.random_state = self.ml_config.get("random_state", 42)
        
        # Initialize base detectors
        self.detectors = {
            'isolation_forest': IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'one_class_svm': OneClassSVM(
                nu=self.contamination,
                gamma='scale'
            ),
            'local_outlier_factor': LocalOutlierFactor(
                contamination=self.contamination,
                novelty=True,
                n_jobs=-1
            )
        }
        
        self.is_fitted = False
        
    def fit(self, X: np.ndarray):
        """Fit all base detectors."""
        logger.debug("Training ensemble detectors...")
        
        for name, detector in self.detectors.items():
            try:
                detector.fit(X)
                logger.debug(f"Trained {name}")
            except Exception as e:
                logger.warning(f"Failed to train {name}: {e}")
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        predictions = []
        
        for name, detector in self.detectors.items():
            try:
                if name == 'isolation_forest':
                    pred = (detector.predict(X) == -1).astype(int)
                elif name == 'local_outlier_factor':
                    pred = (detector.predict(X) == -1).astype(int)
                else:
                    pred = (detector.predict(X) == -1).astype(int)
                
                predictions.append(pred)
                
            except Exception as e:
                logger.warning(f"Prediction failed for {name}: {e}")
                predictions.append(np.zeros(len(X)))
        
        # Ensemble voting (majority vote)
        if predictions:
            ensemble_pred = np.round(np.mean(predictions, axis=0)).astype(int)
        else:
            ensemble_pred = np.zeros(len(X))
        
        return ensemble_pred
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Get ensemble anomaly scores."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before scoring")
        
        scores = []
        
        for name, detector in self.detectors.items():
            try:
                if hasattr(detector, 'score_samples'):
                    score = detector.score_samples(X)
                elif hasattr(detector, 'decision_function'):
                    score = detector.decision_function(X)
                elif hasattr(detector, 'negative_outlier_factor_'):
                    score = detector.negative_outlier_factor_
                else:
                    score = np.zeros(len(X))
                
                scores.append(score)
                
            except Exception as e:
                logger.warning(f"Scoring failed for {name}: {e}")
                scores.append(np.zeros(len(X)))
        
        # Average ensemble scores
        if scores:
            ensemble_scores = np.mean(scores, axis=0)
        else:
            ensemble_scores = np.zeros(len(X))
        
        return ensemble_scores


class FeatureEngineer:
    """Advanced feature engineering for log data."""
    
    def __init__(self, config: Dict):
        """Initialize feature engineer."""
        self.config = config
        self.ml_config = config.get("ml_model", {})
        
        # Text processing parameters
        self.max_features = self.ml_config.get("max_features", 5000)
        self.use_hashing = self.ml_config.get("use_hashing_vectorizer", False)
        self.ngram_range = tuple(self.ml_config.get("ngram_range", [1, 2]))
        
        # Initialize vectorizers
        if self.use_hashing:
            self.text_vectorizer = HashingVectorizer(
                n_features=self.max_features,
                ngram_range=self.ngram_range,
                norm='l2'
            )
        else:
            self.text_vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                stop_words='english',
                min_df=2,
                max_df=0.95
            )
        
        self.feature_cache = {}
        
    def extract_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Extract comprehensive features from log data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        if df.empty:
            return np.array([]).reshape(0, 0), []
        
        logger.debug(f"Extracting features from {len(df)} log entries")
        
        feature_list = []
        feature_names = []
        
        # 1. Text-based features
        if "message" in df.columns:
            text_features, text_names = self._extract_text_features(df["message"])
            if text_features.size > 0:
                feature_list.append(text_features)
                feature_names.extend(text_names)
        
        # 2. Severity features
        if "level" in df.columns:
            severity_features, severity_names = self._extract_severity_features(df["level"])
            feature_list.append(severity_features)
            feature_names.extend(severity_names)
        
        # 3. Temporal features
        if "timestamp" in df.columns:
            temporal_features, temporal_names = self._extract_temporal_features(df["timestamp"])
            if temporal_features.size > 0:
                feature_list.append(temporal_features)
                feature_names.extend(temporal_names)
        
        # 4. Statistical features
        statistical_features, statistical_names = self._extract_statistical_features(df)
        if statistical_features.size > 0:
            feature_list.append(statistical_features)
            feature_names.extend(statistical_names)
        
        # Combine all features
        if feature_list:
            combined_features = np.hstack(feature_list)
        else:
            logger.warning("No features could be extracted")
            return np.array([]).reshape(len(df), 0), []
        
        logger.debug(f"Extracted {combined_features.shape[1]} features")
        return combined_features, feature_names
    
    def _extract_text_features(self, messages: pd.Series) -> Tuple[np.ndarray, List[str]]:
        """Extract TF-IDF features from message text."""
        try:
            # Cache key for this message set
            cache_key = hashlib.md5(''.join(messages.astype(str)).encode()).hexdigest()
            
            if cache_key in self.feature_cache:
                logger.debug("Using cached text features")
                return self.feature_cache[cache_key]
            
            # Clean and preprocess messages
            cleaned_messages = messages.fillna('').astype(str)
            
            # Extract TF-IDF features
            text_matrix = self.text_vectorizer.fit_transform(cleaned_messages)
            
            # Convert to dense array if not too large
            if text_matrix.shape[1] <= 1000:
                text_features = text_matrix.toarray()
            else:
                text_features = text_matrix.toarray()
            
            # Generate feature names
            if hasattr(self.text_vectorizer, 'get_feature_names_out'):
                feature_names = [f"tfidf_{name}" for name in self.text_vectorizer.get_feature_names_out()]
            else:
                feature_names = [f"tfidf_{i}" for i in range(text_features.shape[1])]
            
            # Cache results
            result = (text_features, feature_names)
            self.feature_cache[cache_key] = result
            
            logger.debug(f"Extracted {text_features.shape[1]} text features")
            return result
            
        except Exception as e:
            logger.warning(f"Text feature extraction failed: {e}")
            return np.array([]).reshape(len(messages), 0), []
    
    def _extract_severity_features(self, levels: pd.Series) -> Tuple[np.ndarray, List[str]]:
        """Extract features from log severity levels."""
        severity_map = {
            "CRITICAL": 5, "FATAL": 5,
            "ERROR": 4,
            "WARN": 3, "WARNING": 3,
            "INFO": 2,
            "DEBUG": 1, "TRACE": 1
        }
        
        # Basic severity scores
        severity_scores = levels.str.upper().map(severity_map).fillna(0)
        
        # One-hot encoding for levels
        level_dummies = pd.get_dummies(levels.str.upper(), prefix='level')
        
        # Combine features
        features = np.column_stack([
            severity_scores.values.reshape(-1, 1),
            level_dummies.values
        ])
        
        feature_names = ['severity_score'] + list(level_dummies.columns)
        
        return features, feature_names
    
    def _extract_temporal_features(self, timestamps: pd.Series) -> Tuple[np.ndarray, List[str]]:
        """Extract temporal patterns from timestamps."""
        try:
            # Convert to datetime
            dt_series = pd.to_datetime(timestamps, errors='coerce')
            
            # Skip if no valid timestamps
            if dt_series.isna().all():
                return np.array([]).reshape(len(timestamps), 0), []
            
            # Extract temporal components
            features = []
            feature_names = []
            
            # Hour of day
            hours = dt_series.dt.hour.fillna(0)
            features.append(hours.values.reshape(-1, 1))
            feature_names.append('hour')
            
            # Day of week
            day_of_week = dt_series.dt.dayofweek.fillna(0)
            features.append(day_of_week.values.reshape(-1, 1))
            feature_names.append('day_of_week')
            
            # Hour sin/cos encoding (cyclical)
            hour_sin = np.sin(2 * np.pi * hours / 24)
            hour_cos = np.cos(2 * np.pi * hours / 24)
            features.extend([hour_sin.values.reshape(-1, 1), hour_cos.values.reshape(-1, 1)])
            feature_names.extend(['hour_sin', 'hour_cos'])
            
            # Day sin/cos encoding (cyclical)
            day_sin = np.sin(2 * np.pi * day_of_week / 7)
            day_cos = np.cos(2 * np.pi * day_of_week / 7)
            features.extend([day_sin.values.reshape(-1, 1), day_cos.values.reshape(-1, 1)])
            feature_names.extend(['day_sin', 'day_cos'])
            
            # Time-based anomaly features
            if len(dt_series.dropna()) > 1:
                # Time differences
                time_diffs = dt_series.diff().dt.total_seconds().fillna(0)
                features.append(time_diffs.values.reshape(-1, 1))
                feature_names.append('time_diff_seconds')
                
                # Log frequency in time window
                window_size = pd.Timedelta(minutes=5)
                freq_features = self._calculate_frequency_features(dt_series, window_size)
                features.append(freq_features.reshape(-1, 1))
                feature_names.append('log_frequency_5min')
            
            combined_features = np.hstack(features)
            return combined_features, feature_names
            
        except Exception as e:
            logger.warning(f"Temporal feature extraction failed: {e}")
            return np.array([]).reshape(len(timestamps), 0), []
    
    def _extract_statistical_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Extract statistical features from log data."""
        features = []
        feature_names = []
        
        # Message length features
        if "message" in df.columns:
            message_lengths = df["message"].fillna('').astype(str).str.len()
            features.append(message_lengths.values.reshape(-1, 1))
            feature_names.append('message_length')
            
            # Message complexity (unique character ratio)
            complexity = df["message"].fillna('').apply(
                lambda x: len(set(x)) / max(len(x), 1) if x else 0
            )
            features.append(complexity.values.reshape(-1, 1))
            feature_names.append('message_complexity')
        
        # Pattern-based features
        if "message" in df.columns:
            patterns = {
                'has_numbers': df["message"].str.contains(r'\d', na=False),
                'has_special_chars': df["message"].str.contains(r'[!@#$%^&*()_+\-=\[\]{};:"\\|,.<>\?]', na=False),
                'has_ip_address': df["message"].str.contains(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', na=False),
                'has_url': df["message"].str.contains(r'http[s]?://', na=False),
                'has_email': df["message"].str.contains(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', na=False)
            };
            
            for pattern_name, pattern_series in patterns.items():
                features.append(pattern_series.astype(int).values.reshape(-1, 1))
                feature_names.append(pattern_name)
        
        if features:
            combined_features = np.hstack(features)
            return combined_features, feature_names
        else:
            return np.array([]).reshape(len(df), 0), []
    
    def _calculate_frequency_features(self, timestamps: pd.Series, window: pd.Timedelta) -> np.ndarray:
        """Calculate log frequency in sliding time windows."""
        frequencies = []
        
        for ts in timestamps:
            if pd.isna(ts):
                frequencies.append(0)
                continue
            
            # Count logs in the time window around this timestamp
            window_start = ts - window/2
            window_end = ts + window/2
            
            count = ((timestamps >= window_start) & (timestamps <= window_end)).sum()
            frequencies.append(count)
        
        return np.array(frequencies)


# Backward compatibility - maintain original AnomalyDetector interface
AnomalyDetector = AdvancedAnomalyDetector
