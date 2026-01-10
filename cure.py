CURE
import random
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin_min
import matplotlib.pyplot as plt
# Load dataset
data = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\trainee files\NYC.csv")
# Feature engineering: calculate trip distance
data['trip_distance'] = np.sqrt((data['pickup_longitude'] - data['dropoff_longitude'])
** 2 +
(data['pickup_latitude'] - data['dropoff_latitude']) ** 2)
# Select relevant features and drop NaN values
X = data[['trip_distance', 'trip_duration']].dropna().values
# CURE-like algorithm with shrinking representative points
def cure(X, n_clusters, shrink_factor=0.5, n_representatives=5):
# Randomly select initial centroids
centroids = random.sample(list(X), n_clusters)
clusters = [[] for _ in range(n_clusters)]
# Assign points to nearest centroid
for i, point in enumerate(X):
closest_idx, _ = pairwise_distances_argmin_min([point], centroids)
clusters[closest_idx[0]].append(point)
# Shrink representative points towards centroid
representatives = []
for cluster in clusters:
representatives.append(np.array(random.sample(cluster, min(n_representatives,
len(cluster)))) * shrink_factor)
return representatives, clusters
# Apply CURE approximation
n_clusters = 5 # Adjust as needed
representatives, clusters = cure(X, n_clusters)
# Plotting the results
plt.figure(figsize=(10, 8))
# Plot all data points
for cluster in clusters:
cluster_array = np.array(cluster)
plt.scatter(cluster_array[:, 0], cluster_array[:, 1], alpha=0.5)
# Plot the representative points
for rep in representatives:
plt.scatter(rep[:, 0], rep[:, 1], marker='X', s=200, color='red',
label='Representatives')
plt.title('CURE-like Clustering with Representative Points')
plt.xlabel('Trip Distance')
plt.ylabel('Trip Duration')
plt.legend(loc='upper right')
plt.grid()
plt.show()
