"""
Anomaly Detection using Isolation Forest

This module trains an Isolation Forest on numeric features
to detect anomalous log entries.
"""

import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import Dict

class AnomalyDetector:
    def __init__(self, config: Dict):
        """
        Initializes the anomaly detector.

        Args:
            config (dict): Configuration dictionary from config.yaml
        """
        ml_config = config.get("ml_model", {})
        self.contamination = ml_config.get("contamination", 0.05)
        self.random_state = ml_config.get("random_state", 42)
        self.max_samples = ml_config.get("max_samples", "auto")
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            max_samples=self.max_samples
        )

    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fits Isolation Forest and flags anomalies.

        Args:
            df (pd.DataFrame): DataFrame with log entries

        Returns:
            pd.DataFrame: Same DataFrame with 'is_anomaly' column added
        """
        if df.empty:
            df["is_anomaly"] = []
            return df

        # Simple numeric feature engineering:
        # Count message length and assign severity score
        df["message_length"] = df["message"].apply(lambda x: len(str(x)))
        df["severity"] = df["level"].map({
            "ERROR": 3,
            "WARN": 2,
            "INFO": 1
        }).fillna(0)

        X = df[["message_length", "severity"]]

        # Fit model
        self.model.fit(X)

        # Predict anomalies
        predictions = self.model.predict(X)
        df["is_anomaly"] = (predictions == -1).astype(int)

        return df