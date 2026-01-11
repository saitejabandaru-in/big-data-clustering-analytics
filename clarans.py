CLARANS
import random
import numpy as np
import pandas as pd
from pyclustering.cluster.kmedoids import kmedoids
from sklearn.metrics import pairwise_distances_argmin_min

# Load dataset
data = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\trainee files\NYC.csv")

# Feature engineering: calculate trip distance
data['trip_distance'] = np.sqrt((data['pickup_longitude'] -
data['dropoff_longitude'])**2 +
(data['pickup_latitude'] - data['dropoff_latitude'])**2)

# Select relevant features and drop NaN values
X = data[['trip_distance', 'trip_duration']].dropna().values

# Randomized K-Medoids with local search (CLARANS-like behavior)
def clarans(X, n_clusters, n_local, max_neighbor):
best_medoids = None
best_cost = float('inf')
for _ in range(n_local):

# Randomly initialize medoids
medoids = random.sample(range(len(X)), n_clusters)
kmedoids_instance = kmedoids(X, medoids)
kmedoids_instance.process()

# Compute cost
clusters = kmedoids_instance.get_clusters()
cost = sum(np.linalg.norm(X[point] - X[medoids[i]]) for i, cluster in
enumerate(clusters) for point in cluster)

# Compare cost and update best medoids
if cost < best_cost:
best_cost = cost
best_medoids = medoids
return best_medoids

# Apply CLARANS approximation
n_clusters = 5 # Adjust as needed
best_medoids = clarans(X, n_clusters, n_local=10, max_neighbor=5)
print("Best Medoids:", best_medoids)
