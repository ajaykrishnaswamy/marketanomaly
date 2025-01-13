import numpy as np
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC, Precision, Recall
from sklearn.metrics import f1_score

class AnomalyDetector:
    def __init__(self, model_type='iforest'):
        self.model_type = model_type
        if model_type == 'iforest':
            self.model = IsolationForest(
                contamination=0.1,  # Adjust based on your anomaly percentage
                random_state=42,
                n_estimators=100
            )
        else:  # LSTM
            self.model = self._build_lstm_model()
    
    def _build_lstm_model(self):
        model = Sequential([
            LSTM(128, input_shape=(10, 6), return_sequences=True),
            BatchNormalization(),
            Dropout(0.3),
            
            LSTM(64, return_sequences=True),
            BatchNormalization(),
            Dropout(0.3),
            
            LSTM(32),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            
            Dense(16, activation='relu'),
            
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', AUC(), Precision(), Recall()]
        )
        return model
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if self.model_type == 'iforest':
            self.model.fit(X_train)
        else:  # LSTM
            # Calculate class weights
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(y_train),
                y=y_train
            )
            class_weight_dict = dict(enumerate(class_weights))
            
            # Add callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_auc',
                    patience=10,
                    restore_best_weights=True,
                    mode='max'
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=0.00001
                )
            ]
            
            self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=32,
                class_weight=class_weight_dict,
                callbacks=callbacks
            )
    
    def predict(self, X):
        if self.model_type == 'iforest':
            # Convert iforest predictions from {1, -1} to {0, 1}
            return (self.model.predict(X) == -1).astype(int)
        else:  # LSTM
            return (self.model.predict(X) > 0.5).astype(int)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred)) 
    
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