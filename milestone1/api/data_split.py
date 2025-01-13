import numpy as np
def validate_split(X_train, X_test, y_train, y_test):
    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)
    print("Training labels distribution:", np.unique(y_train, return_counts=True))
    print("Test labels distribution:", np.unique(y_test, return_counts=True)) 