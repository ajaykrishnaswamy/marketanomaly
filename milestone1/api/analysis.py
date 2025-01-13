import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

# Load the data
df = pd.read_csv('FinancialPredictorAnomaly - Sheet1.csv')

# Calculate correlation matrix
correlation_matrix = df[['Y', 'VIX', 'DXY', 'GT10', 'Cl1', 'CRY', 'BDIY']].corr()

# Create correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Features')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# Print basic statistics about anomalies
total_samples = len(df)
anomaly_samples = df['Y'].sum()
print(f"Total samples: {total_samples}")
print(f"Anomaly samples: {anomaly_samples}")
print(f"Anomaly percentage: {(anomaly_samples/total_samples)*100:.2f}%")

def check_label_distribution(y):
    unique, counts = np.unique(y, return_counts=True)
    distribution = dict(zip(unique, counts))
    print("Label distribution:", distribution)
    return distribution

# Add this after loading your data
check_label_distribution(y)  # Add this before model training 