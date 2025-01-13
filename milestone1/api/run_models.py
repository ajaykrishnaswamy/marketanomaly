from data_preprocessing import DataPreprocessor
from models import AnomalyDetector
import pandas as pd
import numpy as np

def check_data_distribution(X_train, X_val, X_test, y_train, y_val, y_test):
    print("\nData Distribution:")
    print(f"Training set - Shape: {X_train.shape}, Class distribution: {np.unique(y_train, return_counts=True)}")
    print(f"Validation set - Shape: {X_val.shape}, Class distribution: {np.unique(y_val, return_counts=True)}")
    print(f"Test set - Shape: {X_test.shape}, Class distribution: {np.unique(y_test, return_counts=True)}")

def main():
    # Initialize preprocessor
    preprocessor = DataPreprocessor('FinancialPredictorAnomaly - Sheet1.csv')
    
    # Load and preprocess data
    X, y = preprocessor.load_and_preprocess()
    
    # Train and evaluate Isolation Forest
    print("\nTraining Isolation Forest...")
    (X_train, X_val, X_test), (y_train, y_val, y_test) = preprocessor.split_data(X, y)
    
    # Check data distribution after splitting
    check_data_distribution(X_train, X_val, X_test, y_train, y_val, y_test)
    
    iforest = AnomalyDetector(model_type='iforest')
    iforest.fit(X_train, y_train)
    print("\nIsolation Forest Results:")
    iforest.evaluate(X_test, y_test)
    
    # Train and evaluate LSTM
    print("\nTraining LSTM...")
    (X_train_lstm, X_val_lstm, X_test_lstm), (y_train_lstm, y_val_lstm, y_test_lstm) = preprocessor.prepare_lstm_data(X, y)
    
    # Check LSTM data distribution
    print("\nLSTM Data Distribution:")
    check_data_distribution(X_train_lstm, X_val_lstm, X_test_lstm, y_train_lstm, y_val_lstm, y_test_lstm)
    
    lstm = AnomalyDetector(model_type='lstm')
    lstm.fit(X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm)
    print("\nLSTM Results:")
    lstm.evaluate(X_test_lstm, y_test_lstm)

    # Save the LSTM model
    lstm.model.save('lstm_model.keras')
    print("LSTM model saved as 'lstm_model.keras'.")

if __name__ == "__main__":
    main() 