"""
Example Custom ML Detector Plugins for LogGuard ML

This module demonstrates how to create custom ML algorithms for anomaly detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import logging

from logguard_ml.plugins import MLDetectorPlugin

logger = logging.getLogger(__name__)


class DBSCANAnomalyDetector(MLDetectorPlugin):
    """Custom DBSCAN-based anomaly detection plugin."""
    
    @property
    def name(self) -> str:
        return "dbscan_detector"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "DBSCAN-based clustering anomaly detector for log analysis"
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.feature_names = []
        
    def initialize(self, config: Dict) -> None:
        """Initialize with configuration."""
        dbscan_config = config.get('dbscan', {})
        self.eps = dbscan_config.get('eps', 0.5)
        self.min_samples = dbscan_config.get('min_samples', 5)
        
        logger.info(f"DBSCAN detector initialized: eps={self.eps}, min_samples={self.min_samples}")
        
    def detect_anomalies(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Detect anomalies using DBSCAN clustering."""
        if not self.validate_data(df):
            raise ValueError("Invalid data format for DBSCAN detector")
        
        # Extract features
        features = self._extract_features(df)
        
        # Fit DBSCAN
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        clusters = self.model.fit_predict(features)
        
        # Anomalies are noise points (cluster label -1)
        anomalies = clusters == -1
        
        # Add results to dataframe
        df = df.copy()
        df['is_anomaly'] = anomalies
        df['anomaly_score'] = np.where(anomalies, -1.0, 0.0)
        df['cluster_id'] = clusters
        
        logger.info(f"DBSCAN detected {anomalies.sum()} anomalies out of {len(df)} logs")
        
        return df
    
    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for clustering."""
        # Text features from messages
        text_features = self.vectorizer.fit_transform(df['message'].fillna(''))
        
        # Numerical features
        numerical_features = []
        
        # Message length
        numerical_features.append(df['message'].str.len().fillna(0))
        
        # Log level encoding
        level_mapping = {'DEBUG': 0, 'INFO': 1, 'WARN': 2, 'ERROR': 3}
        numerical_features.append(df['level'].map(level_mapping).fillna(1))
        
        # Time-based features if timestamp available
        if 'timestamp' in df.columns:
            try:
                timestamps = pd.to_datetime(df['timestamp'])
                numerical_features.append(timestamps.dt.hour)
                numerical_features.append(timestamps.dt.day_of_week)
            except:
                pass
        
        # Combine features
        numerical_array = np.column_stack(numerical_features)
        numerical_array = self.scaler.fit_transform(numerical_array)
        
        # Combine text and numerical features
        combined_features = np.hstack([text_features.toarray(), numerical_array])
        
        self.feature_names = (
            [f'text_feature_{i}' for i in range(text_features.shape[1])] +
            ['message_length', 'log_level', 'hour', 'day_of_week'][:len(numerical_features)]
        )
        
        return combined_features
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importance (not directly available for DBSCAN)."""
        if not self.feature_names:
            return {}
        
        # For DBSCAN, we can't get direct feature importance
        # Return uniform importance for demonstration
        importance = 1.0 / len(self.feature_names)
        return {name: importance for name in self.feature_names}


class RandomForestAnomalyDetector(MLDetectorPlugin):
    """Random Forest-based anomaly detection using supervised learning."""
    
    @property
    def name(self) -> str:
        return "random_forest_detector"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Random Forest anomaly detector with synthetic label generation"
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        self.feature_names = []
        
    def initialize(self, config: Dict) -> None:
        """Initialize with configuration."""
        rf_config = config.get('random_forest', {})
        self.n_estimators = rf_config.get('n_estimators', 100)
        self.contamination = rf_config.get('contamination', 0.1)
        
        logger.info(f"Random Forest detector initialized: n_estimators={self.n_estimators}")
        
    def detect_anomalies(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Detect anomalies using Random Forest with synthetic labels."""
        if not self.validate_data(df):
            raise ValueError("Invalid data format for Random Forest detector")
        
        # Extract features
        features = self._extract_features(df)
        
        # Generate synthetic labels based on heuristics
        labels = self._generate_synthetic_labels(df)
        
        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=42,
            class_weight='balanced'
        )
        self.model.fit(features, labels)
        
        # Predict anomalies
        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)
        
        # Add results to dataframe
        df = df.copy()
        df['is_anomaly'] = predictions == 1
        df['anomaly_score'] = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
        
        logger.info(f"Random Forest detected {predictions.sum()} anomalies out of {len(df)} logs")
        
        return df
    
    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for Random Forest."""
        # Text features
        text_features = self.vectorizer.fit_transform(df['message'].fillna(''))
        
        # Numerical features
        numerical_features = []
        
        # Message length
        numerical_features.append(df['message'].str.len().fillna(0))
        
        # Log level encoding
        level_mapping = {'DEBUG': 0, 'INFO': 1, 'WARN': 2, 'ERROR': 3}
        numerical_features.append(df['level'].map(level_mapping).fillna(1))
        
        # Error patterns in messages
        error_patterns = [
            'error', 'fail', 'exception', 'critical', 'timeout',
            'invalid', 'denied', 'refused', 'abort', 'crash'
        ]
        for pattern in error_patterns:
            numerical_features.append(
                df['message'].str.lower().str.contains(pattern, na=False).astype(int)
            )
        
        # Combine features
        numerical_array = np.column_stack(numerical_features)
        numerical_array = self.scaler.fit_transform(numerical_array)
        
        # Combine text and numerical features  
        combined_features = np.hstack([text_features.toarray(), numerical_array])
        
        self.feature_names = (
            [f'text_feature_{i}' for i in range(text_features.shape[1])] +
            ['message_length', 'log_level'] +
            [f'error_pattern_{pattern}' for pattern in error_patterns]
        )
        
        return combined_features
    
    def _generate_synthetic_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Generate synthetic anomaly labels based on heuristics."""
        labels = np.zeros(len(df))
        
        # Mark ERROR and WARN logs as potential anomalies
        error_mask = df['level'].isin(['ERROR', 'WARN'])
        labels[error_mask] = 1
        
        # Mark very long messages as anomalies
        long_message_mask = df['message'].str.len() > df['message'].str.len().quantile(0.95)
        labels[long_message_mask] = 1
        
        # Mark messages with specific error patterns
        error_patterns = ['exception', 'critical', 'timeout', 'crash', 'fail']
        for pattern in error_patterns:
            pattern_mask = df['message'].str.lower().str.contains(pattern, na=False)
            labels[pattern_mask] = 1
        
        # Ensure we don't have too many anomalies (limit to contamination rate)
        if labels.sum() > len(labels) * self.contamination:
            # Keep only the most confident anomalies
            anomaly_indices = np.where(labels == 1)[0]
            n_keep = int(len(labels) * self.contamination)
            keep_indices = np.random.choice(anomaly_indices, n_keep, replace=False)
            labels[:] = 0
            labels[keep_indices] = 1
        
        return labels
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importance from Random Forest."""
        if self.model is None or not self.feature_names:
            return {}
        
        importance_scores = self.model.feature_importances_
        return dict(zip(self.feature_names, importance_scores))


class StatisticalAnomalyDetector(MLDetectorPlugin):
    """Statistical-based anomaly detection using Z-score and IQR methods."""
    
    @property
    def name(self) -> str:
        return "statistical_detector"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Statistical anomaly detector using Z-score and IQR methods"
    
    def __init__(self):
        self.statistics = {}
        
    def initialize(self, config: Dict) -> None:
        """Initialize with configuration."""
        stats_config = config.get('statistical', {})
        self.z_threshold = stats_config.get('z_threshold', 3.0)
        self.iqr_factor = stats_config.get('iqr_factor', 1.5)
        
        logger.info(f"Statistical detector initialized: z_threshold={self.z_threshold}")
        
    def detect_anomalies(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Detect anomalies using statistical methods."""
        if not self.validate_data(df):
            raise ValueError("Invalid data format for Statistical detector")
        
        df = df.copy()
        
        # Extract numerical features for statistical analysis
        features = self._extract_numerical_features(df)
        
        # Calculate anomaly scores using multiple methods
        z_scores = self._calculate_z_scores(features)
        iqr_scores = self._calculate_iqr_scores(features)
        
        # Combine scores (take maximum anomaly indication)
        combined_scores = np.maximum(z_scores, iqr_scores)
        
        # Determine anomalies
        anomalies = combined_scores > 0.5
        
        df['is_anomaly'] = anomalies
        df['anomaly_score'] = combined_scores
        df['z_score'] = z_scores
        df['iqr_score'] = iqr_scores
        
        logger.info(f"Statistical detector found {anomalies.sum()} anomalies out of {len(df)} logs")
        
        return df
    
    def _extract_numerical_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract numerical features for statistical analysis."""
        features = []
        
        # Message length
        message_lengths = df['message'].str.len().fillna(0)
        features.append(message_lengths)
        
        # Log level as number
        level_mapping = {'DEBUG': 0, 'INFO': 1, 'WARN': 2, 'ERROR': 3}
        level_numeric = df['level'].map(level_mapping).fillna(1)
        features.append(level_numeric)
        
        # Time-based features
        if 'timestamp' in df.columns:
            try:
                timestamps = pd.to_datetime(df['timestamp'])
                features.append(timestamps.dt.hour)
                features.append(timestamps.dt.minute)
            except:
                pass
        
        # Word count in messages
        word_counts = df['message'].str.split().str.len().fillna(0)
        features.append(word_counts)
        
        return np.column_stack(features)
    
    def _calculate_z_scores(self, features: np.ndarray) -> np.ndarray:
        """Calculate Z-score based anomaly scores."""
        z_scores = np.abs((features - np.mean(features, axis=0)) / np.std(features, axis=0))
        
        # Handle division by zero
        z_scores = np.nan_to_num(z_scores)
        
        # Normalize to 0-1 range based on threshold
        z_anomaly_scores = np.max(z_scores / self.z_threshold, axis=1)
        z_anomaly_scores = np.clip(z_anomaly_scores, 0, 1)
        
        return z_anomaly_scores
    
    def _calculate_iqr_scores(self, features: np.ndarray) -> np.ndarray:
        """Calculate IQR-based anomaly scores."""
        iqr_scores = np.zeros(features.shape[0])
        
        for i in range(features.shape[1]):
            feature_col = features[:, i]
            q1 = np.percentile(feature_col, 25)
            q3 = np.percentile(feature_col, 75)
            iqr = q3 - q1
            
            if iqr > 0:
                lower_bound = q1 - self.iqr_factor * iqr
                upper_bound = q3 + self.iqr_factor * iqr
                
                # Calculate how far outside the bounds each point is
                outlier_scores = np.maximum(
                    (lower_bound - feature_col) / iqr,
                    (feature_col - upper_bound) / iqr
                )
                outlier_scores = np.maximum(outlier_scores, 0)
                
                # Take maximum across features
                iqr_scores = np.maximum(iqr_scores, outlier_scores)
        
        # Normalize to 0-1 range
        iqr_scores = np.clip(iqr_scores, 0, 1)
        
        return iqr_scores
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importance based on statistical variance."""
        feature_names = ['message_length', 'log_level', 'hour', 'minute', 'word_count']
        
        # Return uniform importance for statistical features
        importance = 1.0 / len(feature_names)
        return {name: importance for name in feature_names}
