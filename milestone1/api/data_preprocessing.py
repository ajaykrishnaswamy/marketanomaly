import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from typing import Tuple, Union, List

class DataPreprocessor:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.scaler = StandardScaler()
    
    def load_and_preprocess(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and preprocess data with proper time series handling"""
        # Load the data
        df = pd.read_csv(self.data_path)
        
        # Extract features and target
        features = ['VIX', 'DXY', 'GT10', 'Cl1', 'CRY', 'BDIY']
        X = df[features]
        y = df['Y']
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=features)
        
        return X_scaled, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                  train_size: float = 0.8, val_size: float = 0.1) -> Tuple:
        """Split data preserving time series nature"""
        # Calculate split points
        n = len(X)
        train_end = int(n * train_size)
        val_end = int(n * (train_size + val_size))
        
        # Split preserving time order
        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        
        X_val = X.iloc[train_end:val_end]
        y_val = y.iloc[train_end:val_end]
        
        X_test = X.iloc[val_end:]
        y_test = y.iloc[val_end:]
        
        # Apply SMOTE only to training data
        smote = SMOTE(
            random_state=42,
            k_neighbors=min(5, len(y_train[y_train == 1]) - 1)
        )
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        print("\nClass distribution before SMOTE:")
        print(f"Training: {np.unique(y_train, return_counts=True)}")
        print(f"Validation: {np.unique(y_val, return_counts=True)}")
        print(f"Test: {np.unique(y_test, return_counts=True)}")
        
        return (X_train_balanced, X_val, X_test), (y_train_balanced, y_val, y_test)
    
    def prepare_lstm_data(self, X: pd.DataFrame, y: pd.Series, 
                         sequence_length: int = 10) -> Tuple:
        """Prepare data for LSTM with proper sequence handling"""
        # First create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(len(X) - sequence_length):
            X_sequences.append(X.iloc[i:(i + sequence_length)].values)
            y_sequences.append(y.iloc[i + sequence_length])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        # Then split the sequences
        n = len(X_sequences)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)
        
        X_train = X_sequences[:train_end]
        y_train = y_sequences[:train_end]
        
        X_val = X_sequences[train_end:val_end]
        y_val = y_sequences[train_end:val_end]
        
        X_test = X_sequences[val_end:]
        y_test = y_sequences[val_end:]
        
        return (X_train, X_val, X_test), (y_train, y_val, y_test) 