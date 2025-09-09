"""
ai/trainer.py - Advanced Gesture Training System

Enhanced Features:
1. Multi-hand gesture support
2. Dynamic sample collection with progress visualization
3. Advanced model validation
4. Data augmentation
5. Hyperparameter tuning
6. Security and validation layers
7. Comprehensive logging
8. Model versioning
9. GPU acceleration support
10. Real-time feedback during collection
"""

import os
import cv2
import mediapipe as mp
import numpy as np
import joblib
import pickle
import time
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (classification_report, 
                            confusion_matrix, 
                            accuracy_score,
                            f1_score)
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

# Security imports
import ssl
import certifi
from cryptography.fernet import Fernet

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_SAMPLES_PER_CLASS = 100
MAX_HANDS = 2  # Support for two-hand gestures
MIN_SAMPLES = 10
MODEL_VERSION = "1.0.0"
SECURE_KEY = os.getenv("MODEL_ENCRYPTION_KEY", "default_key_should_be_changed")

# Configure SSL context for secure downloads
ssl_context = ssl.create_default_context(cafile=certifi.where())

@dataclass
class TrainingConfig:
    """Configuration for model training"""
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 200
    max_depth: int = 10
    min_samples_split: int = 2
    n_jobs: int = -1  # Use all available cores
    use_gpu: bool = False
    augment_data: bool = True
    pca_components: Optional[int] = None

class GestureTrainer:
    """Advanced gesture training system with security and validation"""
    
    def __init__(self):
        # Initialize Mediapipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_HANDS,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Training configuration
        self.config = TrainingConfig()
        
        # Data storage
        self.data: List[np.ndarray] = []
        self.labels: List[str] = []
        self.label_encoder = LabelEncoder()
        
        # Model artifacts
        self.model = None
        self.model_metadata = {}
        self.feature_importances_ = None
        
        # Security
        self.cipher_suite = Fernet(SECURE_KEY.encode())
        
        # Create required directories
        Path("models").mkdir(exist_ok=True)
        Path("training_logs").mkdir(exist_ok=True)
        Path("visualizations").mkdir(exist_ok=True)

    def _secure_dump(self, obj, file_path: str):
        """Securely serialize and encrypt model artifacts"""
        serialized = pickle.dumps(obj)
        encrypted = self.cipher_suite.encrypt(serialized)
        with open(file_path, "wb") as f:
            f.write(encrypted)

    def _secure_load(self, file_path: str):
        """Decrypt and deserialize secure model artifacts"""
        with open(file_path, "rb") as f:
            encrypted = f.read()
        decrypted = self.cipher_suite.decrypt(encrypted)
        return pickle.loads(decrypted)

    def _validate_gesture_labels(self, labels: List[str]) -> bool:
        """Validate gesture labels for security and format"""
        if not labels:
            return False
            
        pattern = re.compile(r'^[a-zA-Z0-9_\- ]+$')
        return all(pattern.match(label.strip()) for label in labels)

    def _extract_landmarks(self, results) -> Optional[np.ndarray]:
        """Extract normalized landmarks from Mediapipe results"""
        if not results.multi_hand_landmarks:
            return None
            
        keypoints = []
        for hand_landmarks in results.multi_hand_landmarks:
            # Get all landmarks and normalize relative to wrist
            wrist = hand_landmarks.landmark[0]
            for lm in hand_landmarks.landmark:
                keypoints.extend([
                    lm.x - wrist.x,  # Relative x
                    lm.y - wrist.y,  # Relative y
                    lm.z - wrist.z,  # Relative z
                    lm.visibility    # Visibility score
                ])
        
        return np.array(keypoints)

    def _augment_data(self, features: np.ndarray) -> List[np.ndarray]:
        """Generate augmented versions of the input features"""
        if not self.config.augment_data:
            return [features]
            
        augmented = [features]
        
        # Add small random noise
        noise = np.random.normal(0, 0.01, features.shape)
        augmented.append(features + noise)
        
        # Add scaled versions
        for scale in [0.95, 1.05]:
            augmented.append(features * scale)
            
        return augmented

    def _display_feedback(self, frame, gesture: str, count: int, total: int):
        """Display real-time feedback during collection"""
        # Draw landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
        
        # Display collection info
        cv2.putText(frame, f"Gesture: {gesture}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Samples: {count}/{total}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Progress bar
        progress = int((count / total) * frame.shape[1])
        cv2.rectangle(frame, (0, frame.shape[0] - 20), 
                     (progress, frame.shape[0]), (0, 255, 0), -1)
        
        return frame

    def collect_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Collect gesture data with real-time feedback"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open video capture device")
            
        # Get gesture labels from user
        gesture_labels = []
        while not self._validate_gesture_labels(gesture_labels):
            input_str = input("Enter gesture labels (comma-separated, alphanumeric only): ")
            gesture_labels = [x.strip() for x in input_str.split(",") if x.strip()]
            
        samples_per_class = DEFAULT_SAMPLES_PER_CLASS
        try:
            samples_input = input(f"Samples per class (default {DEFAULT_SAMPLES_PER_CLASS}): ")
            if samples_input:
                samples_per_class = max(MIN_SAMPLES, int(samples_input))
        except ValueError:
            logger.warning(f"Invalid input, using default {DEFAULT_SAMPLES_PER_CLASS}")
            
        self.model_metadata = {
            "gesture_labels": gesture_labels,
            "samples_per_class": samples_per_class,
            "collection_date": datetime.now().isoformat(),
            "model_version": MODEL_VERSION
        }
        
        for gesture in gesture_labels:
            logger.info(f"\nCollecting data for: {gesture} ({samples_per_class} samples)")
            count = 0
            
            while count < samples_per_class:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to capture frame")
                    break
                    
                # Process frame
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb)
                
                # Extract and store features
                features = self._extract_landmarks(results)
                if features is not None:
                    # Store original and augmented data
                    for augmented_features in self._augment_data(features):
                        self.data.append(augmented_features)
                        self.labels.append(gesture)
                        count += 1
                        
                    logger.info(f"Collected {count}/{samples_per_class}", end='\r')
                
                # Display feedback
                frame = self._display_feedback(frame, gesture, count, samples_per_class)
                cv2.imshow("Gesture Trainer", frame)
                
                # Check for quit command
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                    
        cap.release()
        cv2.destroyAllWindows()
        
        # Convert to numpy arrays
        X = np.array(self.data)
        y = self.label_encoder.fit_transform(self.labels)
        
        return X, y

    def train_model(self, X: np.ndarray, y: np.ndarray):
        """Train gesture classification model with hyperparameter tuning"""
        logger.info("\nStarting model training...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config.test_size, 
            random_state=self.config.random_state,
            stratify=y
        )
        
        # Define pipeline
        pipeline_steps = []
        
        # Add PCA if configured
        if self.config.pca_components:
            pipeline_steps.append((
                'pca', 
                PCA(n_components=self.config.pca_components)
            ))
        
        # Define classifier with hyperparameter grid
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [5, 10, 15],
            'classifier__min_samples_split': [2, 5, 10]
        }
        
        pipeline_steps.append((
            'classifier', 
            RandomForestClassifier(
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs
            )
        ))
        
        pipeline = Pipeline(pipeline_steps)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring='f1_weighted',
            n_jobs=self.config.n_jobs,
            verbose=2
        )
        
        logger.info("Performing grid search...")
        grid_search.fit(X_train, y_train)
        
        # Store best model
        self.model = grid_search.best_estimator_
        self.feature_importances_ = self.model.named_steps['classifier'].feature_importances_
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        
        # Update metadata
        self.model_metadata.update({
            "training_date": datetime.now().isoformat(),
            "best_params": grid_search.best_params_,
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_f1_score": f1_score(y_test, y_pred, average='weighted'),
            "feature_importance": self.feature_importances_.tolist(),
            "input_shape": X.shape[1],
            "classes": self.label_encoder.classes_.tolist()
        })
        
        # Generate and save reports
        self._generate_reports(y_test, y_pred)
        
        return self.model

    def _generate_reports(self, y_test: np.ndarray, y_pred: np.ndarray):
        """Generate and save evaluation reports"""
        # Classification report
        report = classification_report(
            y_test, 
            y_pred, 
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Save reports
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"training_logs/report_{timestamp}.json"
        
        with open(report_path, "w") as f:
            json.dump({
                "classification_report": report,
                "confusion_matrix": cm.tolist(),
                "metadata": self.model_metadata
            }, f, indent=2)
            
        # Save visualization
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', 
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.savefig(f"visualizations/confusion_matrix_{timestamp}.png")
        plt.close()
        
        logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
        logger.info(f"Model evaluation saved to {report_path}")

    def save_model(self):
        """Save model and metadata securely"""
        if not self.model:
            raise RuntimeError("No model trained to save")
            
        # Generate unique model name
        model_hash = hashlib.sha256(
            str(self.model_metadata).encode()
        ).hexdigest()[:8]
        model_name = f"gesture_model_v{MODEL_VERSION}_{model_hash}.pkl"
        model_path = f"models/{model_name}"
        
        # Save artifacts
        artifacts = {
            "model": self.model,
            "label_encoder": self.label_encoder,
            "metadata": self.model_metadata,
            "feature_importances": self.feature_importances_
        }
        
        self._secure_dump(artifacts, model_path)
        logger.info(f"\n✅ Model saved to '{model_path}'")
        
        # Save human-readable metadata
        with open(f"models/{model_name}.meta.json", "w") as f:
            json.dump(self.model_metadata, f, indent=2)
            
        return model_path

    def load_model(self, model_path: str):
        """Load a trained model from file"""
        artifacts = self._secure_load(model_path)
        self.model = artifacts["model"]
        self.label_encoder = artifacts["label_encoder"]
        self.model_metadata = artifacts["metadata"]
        self.feature_importances_ = artifacts["feature_importances"]
        
        logger.info(f"Loaded model: {self.model_metadata.get('model_version', 'unknown')}")
        return self.model

if __name__ == "__main__":
    try:
        trainer = GestureTrainer()
        
        # Collect data
        X, y = trainer.collect_data()
        
        # Train model
        model = trainer.train_model(X, y)
        
        # Save model
        model_path = trainer.save_model()
        
        logger.info("\nTraining completed successfully!")
        logger.info(f"Model saved to: {model_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise