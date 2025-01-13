import numpy as np
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC, Precision, Recall
from typing import Tuple, Union, List, Dict

class AnomalyDetector:
    def __init__(self, model_type: str = 'iforest'):
        self.model_type = model_type
        if model_type == 'iforest':
            self.model = IsolationForest(
                contamination=0.2,
                random_state=42,
                n_estimators=200,
                max_samples='auto',
                bootstrap=True
            )
        else:  # LSTM
            self.model = self._build_lstm_model()
            
    def _build_lstm_model(self) -> Sequential:
        model = Sequential([
            LSTM(256, input_shape=(10, 6), return_sequences=True),
            BatchNormalization(),
            Dropout(0.3),
            
            LSTM(128, return_sequences=True),
            BatchNormalization(),
            Dropout(0.3),
            
            LSTM(64, return_sequences=False),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(16, activation='relu'),
            BatchNormalization(),
            
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    AUC(name='auc'), 
                    Precision(name='precision'), 
                    Recall(name='recall')]
        )
        return model
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: np.ndarray = None, y_val: np.ndarray = None) -> None:
        if self.model_type == 'iforest':
            self.model.fit(X_train)
        else:  # LSTM
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(y_train),
                y=y_train
            )
            class_weight_dict = dict(enumerate(class_weights))
            
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    mode='min'
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    mode='min',
                    min_lr=0.00001
                )
            ]
            
            self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=32,
                class_weight=class_weight_dict,
                callbacks=callbacks,
                shuffle=False  # Important for time series
            )
    
    def predict(self, X):
        if self.model_type == 'iforest':
            return (self.model.predict(X) == -1).astype(int)
        else:  # LSTM
            return (self.model.predict(X) > 0.5).astype(int)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        
        # Print detailed metrics
        print("\nDetailed Metrics:")
        print("Unique values in predictions:", np.unique(y_pred, return_counts=True))
        print("Unique values in true labels:", np.unique(y_test, return_counts=True))
        
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Calculate metrics with zero_division parameter
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        # Calculate F1 Score manually
        f1 = f1_score(y_test, y_pred)
        print(f"F1 Score: {f1:.2f}")
        
        return y_pred
    
    def tune_threshold(self, X_val, y_val):
        """Find optimal threshold for anomaly detection"""
        predictions = self.model.predict(X_val)
        thresholds = np.arange(0.1, 0.9, 0.1)
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (predictions > threshold).astype(int)
            f1 = f1_score(y_val, y_pred)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                
        return best_threshold 
    
    def evaluate_model(self, y_true, y_pred):
        print("Unique values in predictions:", np.unique(y_pred, return_counts=True))
        print("Unique values in true labels:", np.unique(y_true, return_counts=True))
        
        # Calculate and print confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Then proceed with classification report
        return classification_report(y_true, y_pred, output_dict=False) 