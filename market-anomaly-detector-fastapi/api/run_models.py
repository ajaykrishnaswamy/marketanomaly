from data_preprocessing import DataPreprocessor
from models import AnomalyDetector
import pandas as pd

def main():
    # Initialize preprocessor
    preprocessor = DataPreprocessor('FinancialPredictorAnomaly - Sheet1.csv')
    
    # Load and preprocess data
    X, y = preprocessor.load_and_preprocess()
    
    # Train and evaluate Isolation Forest
    print("\nTraining Isolation Forest...")
    (X_train, X_val, X_test), (y_train, y_val, y_test) = preprocessor.split_data(X, y)
    
    iforest = AnomalyDetector(model_type='iforest')
    iforest.fit(X_train, y_train)
    print("\nIsolation Forest Results:")
    iforest.evaluate(X_test, y_test)
    
    # Train and evaluate LSTM
    print("\nTraining LSTM...")
    X_seq, y_seq = preprocessor.create_sequences(X, y)
    (X_train, X_val, X_test), (y_train, y_val, y_test) = preprocessor.split_data(X_seq, y_seq)
    
    lstm = AnomalyDetector(model_type='lstm')
    lstm.fit(X_train, y_train, X_val, y_val)
    print("\nLSTM Results:")
    lstm.evaluate(X_test, y_test)

    # Save the LSTM model in the recommended format
    lstm.model.save('lstm_model.keras')
    print("LSTM model saved as 'lstm_model.keras'.")

if __name__ == "__main__":
    main() 