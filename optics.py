OPTICS
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS
from sklearn.decomposition import PCA

# Load dataset (only load a subset for testing)
data = pd.read_csv(r"traineefiles\creditcard.csv").sample(10000) # Use 10k samples for quicker results

# Select relevant PCA features for clustering
X = data[["V1", "V2", "V3", "V4", "V5"]].dropna()

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Optional: Reduce dimensionality using PCA to 2D for faster clustering
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Fit OPTICS with adjusted parameters
optics = OPTICS(min_samples=10, xi=0.1, min_cluster_size=0.1) # Adjusted
parameters for performance
labels = optics.fit_predict(X_pca)

# Add labels to the original data
data['OPTICS_Labels'] = labels

# Visualization
plt.figure(figsize=(12, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data['OPTICS_Labels'], cmap='Spectral',
alpha=0.5)
plt.title('OPTICS Clustering on Credit Card Fraud Data (Optimized)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster Label')
plt.grid(True)
plt.show()
# Optional: Identify and visualize outliers
outliers = data[data['OPTICS_Labels'] == -1]
print(f"Number of outliers detected: {len(outliers)}")
