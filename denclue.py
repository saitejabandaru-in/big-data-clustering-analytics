DENCLUE

import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
# Load dataset
data = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\trainee files\creditcard.csv")
# Select relevant PCA features for clustering
X = data[['V1', 'V2', 'V3', 'V4', 'V5']].dropna().values
# Apply Gaussian Kernel Density Estimation for DENCLUE-like clustering
def denclue(X, bandwidth=0.5):
kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(X)
log_density = kde.score_samples(X)
return np.exp(log_density)
# Apply DENCLUE-like density clustering
density = denclue(X)
# Visualize density
plt.scatter(X[:, 0], X[:, 1], c=density, cmap='viridis')
plt.title('DENCLUE-like Density Estimation')
plt.colorbar()
plt.show()