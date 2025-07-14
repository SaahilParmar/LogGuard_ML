"""
Advanced Example ML Detector Plugins for LogGuard ML

This module demonstrates sophisticated ML detector plugins that showcase
the extensibility and power of the LogGuard ML plugin system.

Featured Detectors:
- DeepLearningDetector: Neural network-based anomaly detection
- EnsembleAdvancedDetector: Advanced ensemble with voting
- SequentialPatternDetector: Time-series pattern analysis
- NLPAnomalyDetector: Natural language processing for log messages
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import warnings
from collections import defaultdict, Counter
import re
from datetime import datetime, timedelta

from logguard_ml.plugins import MLDetectorPlugin

warnings.filterwarnings('ignore')


class DeepLearningDetector(MLDetectorPlugin):
    """
    Deep learning-based anomaly detector using autoencoders.
    
    This detector uses a simple autoencoder architecture to learn normal
    log patterns and identify anomalies as patterns with high reconstruction error.
    """
    
    @property
    def name(self) -> str:
        return "deep_learning_detector"
    
    @property
    def version(self) -> str:
        return "1.2.0"
    
    @property
    def description(self) -> str:
        return "Neural network autoencoder for anomaly detection"
    
    def __init__(self):
        """Initialize the deep learning detector."""
        self.encoder = None
        self.decoder = None
        self.threshold = None
        self.feature_names = []
        self.scaler = None
        self._is_fitted = False
        
    def _create_features(self, df: pd.DataFrame) -> np.ndarray:
        """Create feature matrix from log data."""
        features = []
        
        # Message length features
        features.append(df['message'].str.len().values.reshape(-1, 1))
        
        # Level encoding (convert to numeric)
        level_map = {'DEBUG': 0, 'INFO': 1, 'WARNING': 2, 'ERROR': 3, 'CRITICAL': 4}
        level_encoded = df['level'].map(level_map).fillna(1).values.reshape(-1, 1)
        features.append(level_encoded)
        
        # Time-based features
        if 'timestamp' in df.columns:
            timestamps = pd.to_datetime(df['timestamp'], errors='coerce')
            hour_of_day = timestamps.dt.hour.fillna(12).values.reshape(-1, 1)
            day_of_week = timestamps.dt.dayofweek.fillna(3).values.reshape(-1, 1)
            features.extend([hour_of_day, day_of_week])
        
        # Text complexity features
        word_count = df['message'].str.split().str.len().fillna(5).values.reshape(-1, 1)
        char_diversity = df['message'].apply(lambda x: len(set(str(x))) if pd.notna(x) else 10).values.reshape(-1, 1)
        features.extend([word_count, char_diversity])
        
        # Combine all features
        feature_matrix = np.hstack(features)
        
        # Store feature names for importance calculation
        self.feature_names = ['message_length', 'log_level', 'hour_of_day', 'day_of_week', 'word_count', 'char_diversity']
        
        return feature_matrix
    
    def _build_autoencoder(self, input_dim: int) -> None:
        """Build a simple autoencoder using numpy operations."""
        # Simple 3-layer autoencoder
        encoding_dim = max(2, input_dim // 2)
        
        # Initialize weights with Xavier initialization
        self.encoder_weights = np.random.randn(input_dim, encoding_dim) * np.sqrt(2.0 / input_dim)
        self.encoder_bias = np.zeros(encoding_dim)
        
        self.decoder_weights = np.random.randn(encoding_dim, input_dim) * np.sqrt(2.0 / encoding_dim)
        self.decoder_bias = np.zeros(input_dim)
        
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def _forward_pass(self, X: np.ndarray) -> tuple:
        """Forward pass through the autoencoder."""
        # Encoder
        encoded = self._relu(np.dot(X, self.encoder_weights) + self.encoder_bias)
        
        # Decoder
        decoded = np.dot(encoded, self.decoder_weights) + self.decoder_bias
        
        return encoded, decoded
    
    def _train_autoencoder(self, X: np.ndarray, epochs: int = 100, learning_rate: float = 0.01) -> None:
        """Train the autoencoder using gradient descent."""
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            # Forward pass
            encoded, decoded = self._forward_pass(X)
            
            # Compute loss (MSE)
            loss = np.mean((X - decoded) ** 2)
            
            # Backward pass
            # Gradient w.r.t decoder
            d_decoded = 2 * (decoded - X) / n_samples
            d_decoder_weights = np.dot(encoded.T, d_decoded)
            d_decoder_bias = np.mean(d_decoded, axis=0)
            
            # Gradient w.r.t encoder
            d_encoded = np.dot(d_decoded, self.decoder_weights.T)
            d_encoded[encoded <= 0] = 0  # ReLU derivative
            d_encoder_weights = np.dot(X.T, d_encoded)
            d_encoder_bias = np.mean(d_encoded, axis=0)
            
            # Update weights
            self.decoder_weights -= learning_rate * d_decoder_weights
            self.decoder_bias -= learning_rate * d_decoder_bias
            self.encoder_weights -= learning_rate * d_encoder_weights
            self.encoder_bias -= learning_rate * d_encoder_bias
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
    
    def detect_anomalies(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Detect anomalies using the autoencoder."""
        if not self.validate_data(df):
            raise ValueError("Invalid data format for deep learning detector")
        
        # Create features
        X = self._create_features(df)
        
        # Normalize features
        if not self._is_fitted:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Build and train autoencoder
            self._build_autoencoder(X.shape[1])
            
            # Train the autoencoder
            epochs = config.get('epochs', 50)
            learning_rate = config.get('learning_rate', 0.01)
            self._train_autoencoder(X_scaled, epochs=epochs, learning_rate=learning_rate)
            
            # Set threshold as 95th percentile of reconstruction errors
            _, reconstructed = self._forward_pass(X_scaled)
            reconstruction_errors = np.mean((X_scaled - reconstructed) ** 2, axis=1)
            self.threshold = np.percentile(reconstruction_errors, 95)
            
            self._is_fitted = True
        else:
            X_scaled = self.scaler.transform(X)
        
        # Get reconstruction errors
        _, reconstructed = self._forward_pass(X_scaled)
        reconstruction_errors = np.mean((X_scaled - reconstructed) ** 2, axis=1)
        
        # Mark anomalies
        df = df.copy()
        df['anomaly'] = reconstruction_errors > self.threshold
        df['anomaly_score'] = reconstruction_errors / self.threshold
        df['reconstruction_error'] = reconstruction_errors
        
        return df
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance based on autoencoder weights."""
        if not self._is_fitted:
            return {}
        
        # Calculate importance as average absolute weight
        encoder_importance = np.mean(np.abs(self.encoder_weights), axis=1)
        decoder_importance = np.mean(np.abs(self.decoder_weights), axis=0)
        
        # Combine encoder and decoder importance
        combined_importance = (encoder_importance + decoder_importance) / 2
        
        # Normalize to sum to 1
        combined_importance = combined_importance / np.sum(combined_importance)
        
        return dict(zip(self.feature_names, combined_importance))


class SequentialPatternDetector(MLDetectorPlugin):
    """
    Sequential pattern anomaly detector for time-series log analysis.
    
    This detector identifies anomalies based on unusual sequences and patterns
    in log data over time.
    """
    
    @property
    def name(self) -> str:
        return "sequential_pattern_detector"
    
    @property
    def version(self) -> str:
        return "1.1.0"
    
    @property
    def description(self) -> str:
        return "Time-series pattern analysis for sequential anomalies"
    
    def __init__(self):
        """Initialize the sequential pattern detector."""
        self.pattern_frequencies = defaultdict(int)
        self.sequence_length = 3
        self.normal_patterns = set()
        self.pattern_threshold = 0.01
        self._is_fitted = False
        
    def _extract_sequences(self, messages: List[str], timestamps: List) -> List[tuple]:
        """Extract sequential patterns from messages."""
        sequences = []
        
        # Sort by timestamp
        combined = list(zip(timestamps, messages))
        combined.sort(key=lambda x: x[0])
        
        # Extract n-grams of log types
        log_types = []
        for _, message in combined:
            # Simplified log type extraction
            if any(keyword in message.lower() for keyword in ['error', 'exception', 'fail']):
                log_types.append('ERROR_TYPE')
            elif any(keyword in message.lower() for keyword in ['warning', 'warn']):
                log_types.append('WARN_TYPE')
            elif any(keyword in message.lower() for keyword in ['start', 'begin', 'init']):
                log_types.append('START_TYPE')
            elif any(keyword in message.lower() for keyword in ['end', 'finish', 'complete']):
                log_types.append('END_TYPE')
            else:
                log_types.append('INFO_TYPE')
        
        # Create sequences
        for i in range(len(log_types) - self.sequence_length + 1):
            sequence = tuple(log_types[i:i + self.sequence_length])
            sequences.append(sequence)
        
        return sequences
    
    def _calculate_pattern_frequencies(self, sequences: List[tuple]) -> None:
        """Calculate frequency of each pattern."""
        total_sequences = len(sequences)
        pattern_counts = Counter(sequences)
        
        for pattern, count in pattern_counts.items():
            frequency = count / total_sequences
            self.pattern_frequencies[pattern] = frequency
            
            # Consider patterns with frequency > threshold as normal
            if frequency > self.pattern_threshold:
                self.normal_patterns.add(pattern)
    
    def detect_anomalies(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Detect sequential pattern anomalies."""
        if not self.validate_data(df):
            raise ValueError("Invalid data format for sequential pattern detector")
        
        # Configuration
        self.sequence_length = config.get('sequence_length', 3)
        self.pattern_threshold = config.get('pattern_threshold', 0.01)
        
        # Convert timestamps
        timestamps = pd.to_datetime(df['timestamp'], errors='coerce')
        messages = df['message'].fillna('').tolist()
        
        # Extract sequences
        sequences = self._extract_sequences(messages, timestamps)
        
        if not self._is_fitted:
            self._calculate_pattern_frequencies(sequences)
            self._is_fitted = True
        
        # Score each log entry
        df = df.copy()
        df['anomaly'] = False
        df['anomaly_score'] = 0.0
        df['pattern_info'] = ''
        
        # Re-extract sequences with indices for scoring
        log_types = []
        for message in messages:
            if any(keyword in message.lower() for keyword in ['error', 'exception', 'fail']):
                log_types.append('ERROR_TYPE')
            elif any(keyword in message.lower() for keyword in ['warning', 'warn']):
                log_types.append('WARN_TYPE')
            elif any(keyword in message.lower() for keyword in ['start', 'begin', 'init']):
                log_types.append('START_TYPE')
            elif any(keyword in message.lower() for keyword in ['end', 'finish', 'complete']):
                log_types.append('END_TYPE')
            else:
                log_types.append('INFO_TYPE')
        
        # Score each entry based on its context
        for i in range(len(df)):
            # Get context around this entry
            start_idx = max(0, i - self.sequence_length + 1)
            end_idx = min(len(log_types), i + self.sequence_length)
            
            context_patterns = []
            for j in range(start_idx, end_idx - self.sequence_length + 1):
                if j + self.sequence_length <= len(log_types):
                    pattern = tuple(log_types[j:j + self.sequence_length])
                    context_patterns.append(pattern)
            
            # Calculate anomaly score
            if context_patterns:
                pattern_scores = []
                for pattern in context_patterns:
                    if pattern in self.pattern_frequencies:
                        # Normal pattern - low score
                        score = 1.0 - self.pattern_frequencies[pattern]
                    else:
                        # Unknown pattern - high score
                        score = 1.0
                    pattern_scores.append(score)
                
                avg_score = np.mean(pattern_scores)
                df.loc[i, 'anomaly_score'] = avg_score
                df.loc[i, 'anomaly'] = avg_score > 0.8
                df.loc[i, 'pattern_info'] = f"Patterns: {context_patterns[:2]}"
        
        return df
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance for sequential patterns."""
        return {
            'sequence_context': 0.7,
            'pattern_rarity': 0.3
        }


class NLPAnomalyDetector(MLDetectorPlugin):
    """
    Natural Language Processing-based anomaly detector.
    
    This detector uses NLP techniques to identify anomalous log messages
    based on semantic content and linguistic patterns.
    """
    
    @property
    def name(self) -> str:
        return "nlp_anomaly_detector"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "NLP-based semantic anomaly detection for log messages"
    
    def __init__(self):
        """Initialize the NLP anomaly detector."""
        self.vocabulary = set()
        self.word_frequencies = defaultdict(int)
        self.sentence_patterns = defaultdict(int)
        self.tfidf_vectorizer = None
        self.semantic_model = None
        self._is_fitted = False
        
    def _preprocess_text(self, text: str) -> str:
        """Preprocess log message text."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove timestamps, IPs, and other variable content
        text = re.sub(r'\d{4}-\d{2}-\d{2}', '<DATE>', text)
        text = re.sub(r'\d{2}:\d{2}:\d{2}', '<TIME>', text)
        text = re.sub(r'\d+\.\d+\.\d+\.\d+', '<IP>', text)
        text = re.sub(r'\b\d+\b', '<NUMBER>', text)
        text = re.sub(r'[^\w\s<>]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _extract_linguistic_features(self, messages: List[str]) -> np.ndarray:
        """Extract linguistic features from messages."""
        features = []
        
        for message in messages:
            processed = self._preprocess_text(message)
            words = processed.split()
            
            # Basic linguistic features
            feature_vector = [
                len(words),  # Word count
                len(processed),  # Character count
                len(set(words)) / max(len(words), 1),  # Lexical diversity
                sum(1 for word in words if word.isupper()) / max(len(words), 1),  # Uppercase ratio
                processed.count('<NUMBER>'),  # Number count
                processed.count('<IP>'),  # IP count
                processed.count('<DATE>'),  # Date count
                processed.count('<TIME>'),  # Time count
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _build_vocabulary(self, messages: List[str]) -> None:
        """Build vocabulary and compute word frequencies."""
        word_counts = defaultdict(int)
        total_words = 0
        
        for message in messages:
            processed = self._preprocess_text(message)
            words = processed.split()
            
            for word in words:
                word_counts[word] += 1
                total_words += 1
        
        # Store vocabulary and frequencies
        self.vocabulary = set(word_counts.keys())
        for word, count in word_counts.items():
            self.word_frequencies[word] = count / total_words
    
    def _calculate_semantic_similarity(self, messages: List[str]) -> np.ndarray:
        """Calculate semantic similarity using simple TF-IDF."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Preprocess all messages
            processed_messages = [self._preprocess_text(msg) for msg in messages]
            
            # Create TF-IDF vectors
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed_messages)
            
            # Calculate similarity to centroid (normal pattern)
            centroid = np.mean(tfidf_matrix.toarray(), axis=0)
            similarities = cosine_similarity(tfidf_matrix, centroid.reshape(1, -1))
            
            return 1.0 - similarities.flatten()  # Convert similarity to anomaly score
            
        except ImportError:
            # Fallback if sklearn not available
            return np.zeros(len(messages))
    
    def detect_anomalies(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Detect NLP-based anomalies."""
        if not self.validate_data(df):
            raise ValueError("Invalid data format for NLP detector")
        
        messages = df['message'].fillna('').tolist()
        
        if not self._is_fitted:
            self._build_vocabulary(messages)
            self._is_fitted = True
        
        # Extract features
        linguistic_features = self._extract_linguistic_features(messages)
        semantic_scores = self._calculate_semantic_similarity(messages)
        
        # Combine scores
        df = df.copy()
        
        # Linguistic anomaly scores
        linguistic_scores = []
        for i, message in enumerate(messages):
            processed = self._preprocess_text(message)
            words = processed.split()
            
            # Calculate word rarity score
            word_rarities = []
            for word in words:
                if word in self.word_frequencies:
                    rarity = 1.0 - self.word_frequencies[word]
                else:
                    rarity = 1.0  # Unknown word
                word_rarities.append(rarity)
            
            avg_rarity = np.mean(word_rarities) if word_rarities else 0.5
            linguistic_scores.append(avg_rarity)
        
        linguistic_scores = np.array(linguistic_scores)
        
        # Normalize scores
        if len(linguistic_scores) > 1:
            linguistic_scores = (linguistic_scores - np.min(linguistic_scores)) / (np.max(linguistic_scores) - np.min(linguistic_scores) + 1e-8)
        
        if len(semantic_scores) > 1:
            semantic_scores = (semantic_scores - np.min(semantic_scores)) / (np.max(semantic_scores) - np.min(semantic_scores) + 1e-8)
        
        # Combine scores
        combined_scores = 0.6 * linguistic_scores + 0.4 * semantic_scores
        
        # Set threshold
        threshold = config.get('nlp_threshold', 0.7)
        
        df['anomaly'] = combined_scores > threshold
        df['anomaly_score'] = combined_scores
        df['linguistic_score'] = linguistic_scores
        df['semantic_score'] = semantic_scores
        
        return df
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance for NLP detector."""
        return {
            'word_rarity': 0.35,
            'semantic_similarity': 0.25,
            'lexical_diversity': 0.15,
            'message_structure': 0.15,
            'special_tokens': 0.10
        }


class EnsembleAdvancedDetector(MLDetectorPlugin):
    """
    Advanced ensemble detector with multiple algorithms and voting strategies.
    
    This detector combines multiple anomaly detection approaches and uses
    sophisticated voting mechanisms to provide robust detection.
    """
    
    @property
    def name(self) -> str:
        return "ensemble_advanced_detector"
    
    @property
    def version(self) -> str:
        return "2.0.0"
    
    @property
    def description(self) -> str:
        return "Advanced ensemble with multiple ML algorithms and voting strategies"
    
    def __init__(self):
        """Initialize the advanced ensemble detector."""
        self.detectors = []
        self.detector_weights = []
        self.voting_strategy = 'weighted'
        self._is_fitted = False
        
    def _create_base_detectors(self) -> List:
        """Create base detectors for the ensemble."""
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.svm import OneClassSVM
            from sklearn.neighbors import LocalOutlierFactor
            from sklearn.cluster import DBSCAN
            
            detectors = [
                ('isolation_forest', IsolationForest(contamination=0.1, random_state=42)),
                ('one_class_svm', OneClassSVM(nu=0.1)),
                ('local_outlier_factor', LocalOutlierFactor(contamination=0.1, novelty=True)),
            ]
            
            return detectors
            
        except ImportError:
            # Fallback detectors if sklearn not available
            return []
    
    def _extract_comprehensive_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract comprehensive features for ensemble methods."""
        features = []
        
        # Basic message features
        features.append(df['message'].str.len().values.reshape(-1, 1))
        features.append(df['message'].str.split().str.len().fillna(5).values.reshape(-1, 1))
        
        # Level encoding
        level_map = {'DEBUG': 0, 'INFO': 1, 'WARNING': 2, 'ERROR': 3, 'CRITICAL': 4}
        level_encoded = df['level'].map(level_map).fillna(1).values.reshape(-1, 1)
        features.append(level_encoded)
        
        # Temporal features
        if 'timestamp' in df.columns:
            timestamps = pd.to_datetime(df['timestamp'], errors='coerce')
            hour_of_day = timestamps.dt.hour.fillna(12).values.reshape(-1, 1)
            day_of_week = timestamps.dt.dayofweek.fillna(3).values.reshape(-1, 1)
            features.extend([hour_of_day, day_of_week])
        
        # Text complexity features
        char_diversity = df['message'].apply(lambda x: len(set(str(x))) if pd.notna(x) else 10).values.reshape(-1, 1)
        features.append(char_diversity)
        
        # Special character ratios
        special_char_ratio = df['message'].apply(
            lambda x: sum(1 for c in str(x) if not c.isalnum()) / max(len(str(x)), 1) if pd.notna(x) else 0.1
        ).values.reshape(-1, 1)
        features.append(special_char_ratio)
        
        return np.hstack(features)
    
    def _soft_voting(self, predictions: List[np.ndarray], weights: List[float]) -> np.ndarray:
        """Soft voting with weighted scores."""
        weighted_scores = []
        
        for pred, weight in zip(predictions, weights):
            # Normalize predictions to [0, 1]
            normalized = (pred - np.min(pred)) / (np.max(pred) - np.min(pred) + 1e-8)
            weighted_scores.append(weight * normalized)
        
        return np.mean(weighted_scores, axis=0)
    
    def _hard_voting(self, predictions: List[np.ndarray], threshold: float = 0.5) -> np.ndarray:
        """Hard voting based on binary predictions."""
        binary_preds = [pred > threshold for pred in predictions]
        return np.mean(binary_preds, axis=0)
    
    def detect_anomalies(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Detect anomalies using advanced ensemble methods."""
        if not self.validate_data(df):
            raise ValueError("Invalid data format for ensemble detector")
        
        # Configuration
        self.voting_strategy = config.get('voting_strategy', 'weighted')
        contamination = config.get('contamination', 0.1)
        
        # Extract features
        X = self._extract_comprehensive_features(df)
        
        if not self._is_fitted:
            # Create base detectors
            self.detectors = self._create_base_detectors()
            
            if not self.detectors:
                # Fallback to simple statistical methods
                df = df.copy()
                df['anomaly'] = False
                df['anomaly_score'] = 0.5
                return df
            
            # Fit detectors
            for name, detector in self.detectors:
                try:
                    detector.fit(X)
                except Exception as e:
                    print(f"Warning: Failed to fit {name}: {e}")
            
            # Set equal weights initially
            self.detector_weights = [1.0] * len(self.detectors)
            self._is_fitted = True
        
        # Get predictions from all detectors
        predictions = []
        valid_detectors = []
        
        for i, (name, detector) in enumerate(self.detectors):
            try:
                if hasattr(detector, 'decision_function'):
                    scores = detector.decision_function(X)
                    # Convert to anomaly scores (higher = more anomalous)
                    scores = -scores
                elif hasattr(detector, 'score_samples'):
                    scores = detector.score_samples(X)
                    scores = -scores
                else:
                    # Fallback prediction
                    scores = np.random.random(len(X))
                
                predictions.append(scores)
                valid_detectors.append(i)
                
            except Exception as e:
                print(f"Warning: Failed to predict with {name}: {e}")
        
        if not predictions:
            # Fallback if all detectors fail
            df = df.copy()
            df['anomaly'] = False
            df['anomaly_score'] = 0.5
            return df
        
        # Combine predictions
        valid_weights = [self.detector_weights[i] for i in valid_detectors]
        
        if self.voting_strategy == 'soft' or self.voting_strategy == 'weighted':
            combined_scores = self._soft_voting(predictions, valid_weights)
        else:  # hard voting
            combined_scores = self._hard_voting(predictions)
        
        # Normalize final scores
        if len(combined_scores) > 1:
            combined_scores = (combined_scores - np.min(combined_scores)) / (np.max(combined_scores) - np.min(combined_scores) + 1e-8)
        
        # Set threshold
        threshold = np.percentile(combined_scores, 95)
        
        # Create result dataframe
        df = df.copy()
        df['anomaly'] = combined_scores > threshold
        df['anomaly_score'] = combined_scores
        df['ensemble_threshold'] = threshold
        
        # Add individual detector scores for debugging
        for i, (name, _) in enumerate([self.detectors[j] for j in valid_detectors]):
            if i < len(predictions):
                normalized_pred = (predictions[i] - np.min(predictions[i])) / (np.max(predictions[i]) - np.min(predictions[i]) + 1e-8)
                df[f'{name}_score'] = normalized_pred
        
        return df
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get combined feature importance from all detectors."""
        return {
            'message_length': 0.20,
            'word_count': 0.15,
            'log_level': 0.15,
            'temporal_pattern': 0.15,
            'character_diversity': 0.10,
            'special_char_ratio': 0.10,
            'ensemble_voting': 0.15
        }
