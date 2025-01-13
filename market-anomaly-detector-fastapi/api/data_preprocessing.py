import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.scaler = StandardScaler()
        
    def balance_data(self, X, y):
        """Balance the dataset using SMOTE"""
        print("Original class distribution:", Counter(y))
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        print("Balanced class distribution:", Counter(y_balanced))
        return X_balanced, y_balanced
        
    def load_and_preprocess(self):
        # Load the data
        df = pd.read_csv(self.data_path)
        
        # Extract features and target
        features = ['VIX', 'DXY', 'GT10', 'Cl1', 'CRY', 'BDIY']
        X = df[features]
        y = df['Y']
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=features)
        
        # Add balancing step
        X_balanced, y_balanced = self.balance_data(X_scaled, y)
        
        return X_balanced, y_balanced
    
    def create_sequences(self, X, y, sequence_length=10):
        """Create sequences for LSTM"""
        X_seq, y_seq = [], []
        for i in range(len(X) - sequence_length):
            X_seq.append(X.iloc[i:(i + sequence_length)].values)
            y_seq.append(y.iloc[i + sequence_length])
        return np.array(X_seq), np.array(y_seq)
    
    def split_data(self, X, y, train_size=0.7, val_size=0.15):
        """Split data maintaining temporal order"""
        total_size = len(X)
        train_end = int(total_size * train_size)
        val_end = int(total_size * (train_size + val_size))
        
        if isinstance(X, pd.DataFrame):
            X_train = X.iloc[:train_end]
            X_val = X.iloc[train_end:val_end]
            X_test = X.iloc[val_end:]
        else:  # For LSTM sequences (numpy arrays)
            X_train = X[:train_end]
            X_val = X[train_end:val_end]
            X_test = X[val_end:]
            
        y_train = y[:train_end]
        y_val = y[train_end:val_end]
        y_test = y[val_end:]
        
        return (X_train, X_val, X_test), (y_train, y_val, y_test) 
    
    def engineer_features(self, df):
        """Add engineered features"""
        # Rolling statistics
        window_sizes = [5, 10, 20]
        features = ['VIX', 'DXY', 'GT10', 'Cl1', 'CRY', 'BDIY']
        
        for feature in features:
            for window in window_sizes:
                # Rolling mean
                df[f'{feature}_rolling_mean_{window}'] = df[feature].rolling(window=window).mean()
                # Rolling std
                df[f'{feature}_rolling_std_{window}'] = df[feature].rolling(window=window).std()
                # Rolling min/max spread
                df[f'{feature}_rolling_spread_{window}'] = (
                    df[feature].rolling(window=window).max() - 
                    df[feature].rolling(window=window).min()
                )
        
        # Percentage changes
        for feature in features:
            df[f'{feature}_pct_change'] = df[feature].pct_change()
            
        # Cross-feature ratios
        df['VIX_DXY_ratio'] = df['VIX'] / df['DXY']
        df['GT10_CL1_ratio'] = df['GT10'] / df['Cl1']
        
        # Fill NaN values created by rolling windows
        df.fillna(method='bfill', inplace=True)
        
        return df 
    
    def time_series_cv_split(self, X, y, n_splits=5):
        """Implement time series cross-validation"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train and evaluate model
            model = self.train_model(X_train, y_train)
            score = self.evaluate_model(model, X_val, y_val)
            cv_scores.append(score)
            
        return np.mean(cv_scores), np.std(cv_scores) 
    
    def train_model(self, X, y):
        # Define the LSTM model
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))  # For binary classification

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Fit the model
        model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2)
        
        return model 