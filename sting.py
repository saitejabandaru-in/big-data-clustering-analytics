STING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Load dataset
data = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\trainee files\NYC.csv")
# Feature engineering: extract coordinates
X = data[['pickup_longitude', 'pickup_latitude']].dropna().values
# STING-like grid-based clustering algorithm
def sting_grid_clustering(X, grid_size):
grid = {}
# Create grid
for point in X:
grid_idx = tuple(np.floor(point / grid_size))
if grid_idx not in grid:
grid[grid_idx] = []
grid[grid_idx].append(point)
# Return grid clusters
return grid
# Apply STING-like clustering
grid_size = 0.01 # Adjust grid size for spatial clustering
grid_clusters = sting_grid_clustering(X, grid_size)
# Visualize grid cells
plt.scatter(X[:, 0], X[:, 1], s=5)
plt.title('STING-like Grid Clustering for NYC Taxi Data')
plt.grid(True)
plt.show()