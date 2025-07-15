"""
Advanced AI/ML Enhancements for LogGuard ML

This module provides cutting-edge machine learning capabilities including:
- Transformer-based log analysis
- Real-time adaptive learning
- Multi-modal anomaly detection
- Explainable AI (XAI) features
- AutoML capabilities
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import pickle
from pathlib import Path
import json
import time

# Advanced ML imports
try:
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available. Install with: pip install torch transformers")

try:
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    import optuna
    AUTOML_AVAILABLE = True
except ImportError:
    AUTOML_AVAILABLE = False
    logging.warning("AutoML dependencies not available")

try:
    import shap
    import lime
    from lime.lime_text import LimeTextExplainer
    EXPLAINABILITY_AVAILABLE = True
except ImportError:
    EXPLAINABILITY_AVAILABLE = False
    logging.warning("Explainability libraries not available. Install with: pip install shap lime")

logger = logging.getLogger(__name__)


class TransformerLogAnalyzer:
    """
    Transformer-based log analysis using pre-trained language models
    for semantic understanding of log messages.
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        """
        Initialize transformer-based analyzer.
        
        Args:
            model_name: HuggingFace model name
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library required for this feature")
        
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()
    
    def _load_model(self):
        """Load the transformer model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Loaded transformer model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load transformer model: {e}")
            raise
    
    def encode_log_messages(self, messages: List[str], max_length: int = 512) -> np.ndarray:
        """
        Encode log messages into embeddings using transformer model.
        
        Args:
            messages: List of log messages
            max_length: Maximum sequence length
            
        Returns:
            Array of embeddings
        """
        embeddings = []
        
        for message in messages:
            # Tokenize and encode
            inputs = self.tokenizer(
                message,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling of last hidden states
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                embeddings.append(embedding[0])
        
        return np.array(embeddings)
    
    def semantic_similarity(self, message1: str, message2: str) -> float:
        """
        Calculate semantic similarity between two log messages.
        
        Args:
            message1: First log message
            message2: Second log message
            
        Returns:
            Similarity score (0-1)
        """
        embeddings = self.encode_log_messages([message1, message2])
        
        # Calculate cosine similarity
        dot_product = np.dot(embeddings[0], embeddings[1])
        norm_product = np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        
        return dot_product / norm_product if norm_product > 0 else 0.0


class AdaptiveLearningDetector:
    """
    Adaptive anomaly detector that learns from new data in real-time
    and updates its understanding of normal vs anomalous patterns.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize adaptive learning detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.base_model = None
        self.adaptive_threshold = 0.5
        self.learning_rate = config.get("adaptive_learning_rate", 0.01)
        self.update_frequency = config.get("update_frequency", 100)  # samples
        self.feedback_buffer = []
        self.model_performance_history = []
        
    def update_with_feedback(self, X: np.ndarray, y: np.ndarray, feedback: List[bool]):
        """
        Update model with human feedback on predictions.
        
        Args:
            X: Feature matrix
            y: Predictions
            feedback: Human feedback (True = correct, False = incorrect)
        """
        # Store feedback for batch learning
        for i, is_correct in enumerate(feedback):
            self.feedback_buffer.append({
                'features': X[i],
                'prediction': y[i],
                'correct': is_correct,
                'timestamp': time.time()
            })
        
        # Update model if buffer is full
        if len(self.feedback_buffer) >= self.update_frequency:
            self._retrain_with_feedback()
    
    def _retrain_with_feedback(self):
        """Retrain model using accumulated feedback."""
        if not self.feedback_buffer:
            return
        
        # Prepare training data from feedback
        X_feedback = np.array([item['features'] for item in self.feedback_buffer])
        y_feedback = np.array([
            1 if item['correct'] else -1 
            for item in self.feedback_buffer
        ])
        
        # Retrain model (simplified - would use more sophisticated approach)
        if self.base_model is not None:
            # Update model with new data
            self.base_model.fit(X_feedback, y_feedback)
        
        # Clear buffer
        self.feedback_buffer = []
        
        logger.info("Model updated with feedback data")


class MultiModalAnomalyDetector:
    """
    Multi-modal anomaly detection combining text, temporal, and statistical features.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize multi-modal detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.text_analyzer = None
        self.temporal_detector = None
        self.statistical_detector = None
        
        if TRANSFORMERS_AVAILABLE:
            self.text_analyzer = TransformerLogAnalyzer()
    
    def extract_multimodal_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Extract features from multiple modalities.
        
        Args:
            df: DataFrame with log data
            
        Returns:
            Dictionary of feature arrays by modality
        """
        features = {}
        
        # Text features using transformers
        if self.text_analyzer and 'message' in df.columns:
            features['text'] = self.text_analyzer.encode_log_messages(
                df['message'].tolist()
            )
        
        # Temporal features
        if 'timestamp' in df.columns:
            features['temporal'] = self._extract_temporal_features(df)
        
        # Statistical features
        features['statistical'] = self._extract_statistical_features(df)
        
        return features
    
    def _extract_temporal_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract temporal features from timestamps."""
        temporal_features = []
        
        # Convert timestamps
        timestamps = pd.to_datetime(df['timestamp'])
        
        for ts in timestamps:
            features = [
                ts.hour,
                ts.minute,
                ts.day,
                ts.weekday(),
                ts.month
            ]
            temporal_features.append(features)
        
        return np.array(temporal_features)
    
    def _extract_statistical_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract statistical features from log data."""
        statistical_features = []
        
        for _, row in df.iterrows():
            features = [
                len(row.get('message', '')),
                row.get('message', '').count(' '),
                row.get('message', '').count('ERROR'),
                row.get('message', '').count('WARNING'),
                row.get('message', '').count('INFO'),
            ]
            statistical_features.append(features)
        
        return np.array(statistical_features)


class ExplainableAnomalyDetector:
    """
    Provides explainable AI capabilities for anomaly detection results.
    """
    
    def __init__(self, base_detector):
        """
        Initialize explainable detector.
        
        Args:
            base_detector: Base anomaly detection model
        """
        self.base_detector = base_detector
        self.explainer = None
        
        if EXPLAINABILITY_AVAILABLE:
            self.lime_explainer = LimeTextExplainer(class_names=['Normal', 'Anomaly'])
    
    def explain_prediction(self, log_message: str, features: np.ndarray) -> Dict:
        """
        Explain why a log message was classified as anomalous.
        
        Args:
            log_message: Original log message
            features: Feature vector used for prediction
            
        Returns:
            Explanation dictionary
        """
        explanation = {
            'message': log_message,
            'prediction': None,
            'confidence': None,
            'feature_importance': None,
            'text_explanation': None
        }
        
        # Get prediction and confidence
        prediction = self.base_detector.predict([features])[0]
        confidence = getattr(self.base_detector, 'decision_function', lambda x: [0])([features])[0]
        
        explanation['prediction'] = 'Anomaly' if prediction == -1 else 'Normal'
        explanation['confidence'] = abs(confidence)
        
        # SHAP explanation for feature importance
        if EXPLAINABILITY_AVAILABLE and hasattr(self.base_detector, 'predict_proba'):
            try:
                explainer = shap.Explainer(self.base_detector)
                shap_values = explainer([features])
                explanation['feature_importance'] = shap_values.values[0].tolist()
            except Exception as e:
                logger.warning(f"SHAP explanation failed: {e}")
        
        # LIME explanation for text
        if EXPLAINABILITY_AVAILABLE and self.lime_explainer:
            try:
                def predict_fn(texts):
                    # Simplified prediction function for LIME
                    return np.array([[0.5, 0.5] for _ in texts])
                
                lime_explanation = self.lime_explainer.explain_instance(
                    log_message, predict_fn, num_features=10
                )
                explanation['text_explanation'] = lime_explanation.as_list()
            except Exception as e:
                logger.warning(f"LIME explanation failed: {e}")
        
        return explanation


class AutoMLAnomalyDetector:
    """
    Automated machine learning for anomaly detection with hyperparameter optimization.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize AutoML detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.best_model = None
        self.best_params = None
        self.optimization_study = None
        
        if not AUTOML_AVAILABLE:
            logger.warning("AutoML features not available")
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Best hyperparameters
        """
        if not AUTOML_AVAILABLE:
            raise ImportError("AutoML dependencies required")
        
        def objective(trial):
            # Define hyperparameter search space
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            }
            
            # Train model with suggested parameters
            model = GradientBoostingClassifier(**params, random_state=42)
            
            # Cross-validation score
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(model, X, y, cv=3, scoring='f1')
            return scores.mean()
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)
        
        self.optimization_study = study
        self.best_params = study.best_params
        
        # Train final model with best parameters
        self.best_model = GradientBoostingClassifier(**self.best_params, random_state=42)
        self.best_model.fit(X, y)
        
        return self.best_params
    
    def get_optimization_insights(self) -> Dict:
        """Get insights from hyperparameter optimization."""
        if not self.optimization_study:
            return {}
        
        return {
            'best_value': self.optimization_study.best_value,
            'best_params': self.optimization_study.best_params,
            'n_trials': len(self.optimization_study.trials),
            'optimization_history': [
                {'trial': i, 'value': trial.value}
                for i, trial in enumerate(self.optimization_study.trials)
            ]
        }


class RealTimeMLPipeline:
    """
    Real-time ML pipeline for streaming log analysis.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize real-time ML pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model = None
        self.feature_buffer = []
        self.prediction_buffer = []
        self.buffer_size = config.get("buffer_size", 1000)
        self.batch_size = config.get("batch_size", 100)
    
    def process_stream(self, log_entry: Dict) -> Dict:
        """
        Process a single log entry in real-time.
        
        Args:
            log_entry: Log entry dictionary
            
        Returns:
            Processing result with prediction
        """
        # Extract features
        features = self._extract_features(log_entry)
        
        # Make prediction
        if self.model:
            prediction = self.model.predict([features])[0]
            confidence = getattr(self.model, 'decision_function', lambda x: [0])([features])[0]
        else:
            prediction = 0
            confidence = 0.0
        
        # Buffer for batch retraining
        self.feature_buffer.append(features)
        self.prediction_buffer.append(prediction)
        
        # Trigger retraining if buffer is full
        if len(self.feature_buffer) >= self.batch_size:
            self._retrain_model()
        
        return {
            'log_entry': log_entry,
            'prediction': 'Anomaly' if prediction == -1 else 'Normal',
            'confidence': abs(confidence),
            'timestamp': time.time()
        }
    
    def _extract_features(self, log_entry: Dict) -> np.ndarray:
        """Extract features from log entry."""
        # Simplified feature extraction
        message = log_entry.get('message', '')
        return np.array([
            len(message),
            message.count('ERROR'),
            message.count('WARNING'),
            message.count('INFO'),
            len(message.split())
        ])
    
    def _retrain_model(self):
        """Retrain model with buffered data."""
        if len(self.feature_buffer) < self.batch_size:
            return
        
        # Simplified retraining logic
        X = np.array(self.feature_buffer[-self.batch_size:])
        
        if self.model is None:
            from sklearn.ensemble import IsolationForest
            self.model = IsolationForest(contamination=0.1, random_state=42)
        
        self.model.fit(X)
        
        logger.info(f"Model retrained with {len(X)} samples")
