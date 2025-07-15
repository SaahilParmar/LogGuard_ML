"""
Anomaly Detection Module for LogGuard ML

This module provides machine learning-based anomaly detection capabilities
using various algorithms including Isolation Forest.

Classes:
    AnomalyDetector: Main class for detecting anomalies in log data
    AnomalyDetectionError: Custom exception for anomaly detection errors

Example:
    >>> from logguard_ml.core.ml_model import AnomalyDetector
    >>> config = {"ml_model": {"contamination": 0.05}}
    >>> detector = AnomalyDetector(config)
    >>> df_with_anomalies = detector.detect_anomalies(df)
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class AnomalyDetectionError(Exception):
    """Custom exception for anomaly detection errors."""

    pass


class AnomalyDetector:
    """
    Machine learning-based anomaly detector for log entries.

    The AnomalyDetector uses Isolation Forest algorithm to identify
    unusual patterns in log data based on various features including
    message content, severity levels, and temporal patterns.

    Attributes:
        model: Trained Isolation Forest model
        scaler: StandardScaler for feature normalization
        vectorizer: TF-IDF vectorizer for text features
        config: Configuration dictionary

    Example:
        >>> config = {
        ...     "ml_model": {
        ...         "contamination": 0.05,
        ...         "random_state": 42,
        ...         "max_samples": "auto"
        ...     }
        ... }
        >>> detector = AnomalyDetector(config)
        >>> anomalies_df = detector.detect_anomalies(log_df)
    """

    def __init__(self, config: Dict) -> None:
        """
        Initialize the anomaly detector with configuration.

        Args:
            config: Configuration dictionary containing ML model parameters

        Raises:
            AnomalyDetectionError: If configuration is invalid
        """
        if not isinstance(config, dict):
            raise AnomalyDetectionError("Configuration must be a dictionary")

        self.config = config
        ml_config = config.get("ml_model", {})

        # Model parameters with validation
        self.contamination = self._validate_contamination(
            ml_config.get("contamination", 0.05)
        )
        self.random_state = ml_config.get("random_state", 42)
        self.max_samples = ml_config.get("max_samples", "auto")

        # Initialize components
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            max_samples=self.max_samples,
            n_estimators=100,
        )
        self.scaler = StandardScaler()
        self.vectorizer = TfidfVectorizer(
            max_features=100, stop_words="english", ngram_range=(1, 2)
        )

        self._is_fitted = False
        logger.info(
            f"AnomalyDetector initialized with contamination={self.contamination}"
        )

    def _validate_contamination(self, contamination: Union[float, str]) -> float:
        """
        Validate contamination parameter.

        Args:
            contamination: Contamination parameter value

        Returns:
            Valid contamination value

        Raises:
            AnomalyDetectionError: If contamination is invalid
        """
        if isinstance(contamination, str) and contamination == "auto":
            return "auto"

        try:
            contamination = float(contamination)
            if not 0 < contamination < 0.5:
                raise ValueError("Contamination must be between 0 and 0.5")
            return contamination
        except (ValueError, TypeError) as e:
            raise AnomalyDetectionError(f"Invalid contamination parameter: {e}")

    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies in log DataFrame.

        Args:
            df: DataFrame containing log entries with columns like
               'message', 'level', 'timestamp'

        Returns:
            DataFrame with additional columns:
            - 'is_anomaly': Binary flag (1 for anomaly, 0 for normal)
            - 'anomaly_score': Anomaly score from the model
            - 'message_length': Length of log message
            - 'severity_score': Numeric severity score

        Raises:
            AnomalyDetectionError: If detection fails or data is invalid
        """
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return self._add_empty_anomaly_columns(df)

        logger.info(f"Detecting anomalies in {len(df)} log entries")

        try:
            # Create a copy to avoid modifying original
            result_df = df.copy()

            # Extract features
            features = self._extract_features(result_df)

            if features.shape[1] == 0:
                logger.warning("No features could be extracted")
                return self._add_empty_anomaly_columns(result_df)

            # Fit and predict
            self.model.fit(features)
            predictions = self.model.predict(features)
            scores = self.model.score_samples(features)

            # Add results to DataFrame
            result_df["is_anomaly"] = (predictions == -1).astype(int)
            result_df["anomaly_score"] = scores

            self._is_fitted = True

            anomaly_count = result_df["is_anomaly"].sum()
            logger.info(
                f"Detected {anomaly_count} anomalies ({anomaly_count/len(df)*100:.2f}%)"
            )

            return result_df

        except Exception as e:
            raise AnomalyDetectionError(f"Anomaly detection failed: {e}")

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract features from log DataFrame for anomaly detection.

        Args:
            df: Input DataFrame

        Returns:
            Feature matrix for ML model
        """
        features_list = []

        # Text-based features
        if "message" in df.columns:
            # Message length
            df["message_length"] = df["message"].astype(str).str.len()
            features_list.append(df[["message_length"]])

            # TF-IDF features (limited to avoid memory issues)
            try:
                messages = df["message"].astype(str).fillna("")
                if len(messages) > 1:  # Need at least 2 samples for TF-IDF
                    tfidf_features = self.vectorizer.fit_transform(messages)
                    tfidf_df = pd.DataFrame(
                        tfidf_features.toarray(),
                        columns=[f"tfidf_{i}" for i in range(tfidf_features.shape[1])],
                    )
                    features_list.append(tfidf_df)
            except Exception as e:
                logger.warning(f"Could not extract TF-IDF features: {e}")

        # Severity-based features
        if "level" in df.columns:
            severity_map = {
                "CRITICAL": 5,
                "ERROR": 4,
                "WARN": 3,
                "WARNING": 3,
                "INFO": 2,
                "DEBUG": 1,
            }
            df["severity_score"] = df["level"].str.upper().map(severity_map).fillna(0)
            features_list.append(df[["severity_score"]])

        # Temporal features
        if "timestamp" in df.columns:
            try:
                timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
                if not timestamps.isna().all():
                    df["hour"] = timestamps.dt.hour
                    df["day_of_week"] = timestamps.dt.dayofweek
                    features_list.append(df[["hour", "day_of_week"]])
            except Exception as e:
                logger.warning(f"Could not extract temporal features: {e}")

        if not features_list:
            logger.warning("No features could be extracted")
            return np.array([]).reshape(len(df), 0)

        # Combine all features
        combined_features = pd.concat(features_list, axis=1)

        # Handle missing values
        combined_features = combined_features.fillna(0)

        # Scale numerical features
        feature_array = self.scaler.fit_transform(combined_features)

        logger.debug(f"Extracted {feature_array.shape[1]} features")
        return feature_array

    def _add_empty_anomaly_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add empty anomaly columns to DataFrame."""
        result_df = df.copy()
        result_df["is_anomaly"] = 0
        result_df["anomaly_score"] = 0.0
        return result_df

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance from the trained model.

        Returns:
            Dictionary of feature names and their importance scores,
            or None if model is not fitted
        """
        if not self._is_fitted:
            logger.warning("Model not fitted yet")
            return None

        # For Isolation Forest, we can't get direct feature importance
        # This is a placeholder for future enhancement
        return {"note": "Feature importance not available for Isolation Forest"}

    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.

        Args:
            filepath: Path to save the model

        Raises:
            AnomalyDetectionError: If model saving fails
        """
        if not self._is_fitted:
            raise AnomalyDetectionError("Model must be fitted before saving")

        try:
            import joblib

            model_data = {
                "model": self.model,
                "scaler": self.scaler,
                "vectorizer": self.vectorizer,
                "config": self.config,
            }
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            raise AnomalyDetectionError(f"Failed to save model: {e}")

    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.

        Args:
            filepath: Path to the saved model

        Raises:
            AnomalyDetectionError: If model loading fails
        """
        try:
            import joblib

            model_data = joblib.load(filepath)
            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.vectorizer = model_data["vectorizer"]
            self._is_fitted = True
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            raise AnomalyDetectionError(f"Failed to load model: {e}")
