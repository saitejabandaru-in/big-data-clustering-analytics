ARI COMPARISON
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import MiniBatchKMeans, Birch, OPTICS
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
import random

# Load dataset (for demonstration, use NYC dataset)
data = pd.read_csv(r"trainee files\NYC.csv")

# Create synthetic true labels (for demonstration purposes)
np.random.seed(42)
data['true_labels'] = np.random.randint(0, 5, data.shape[0])

# Feature engineering: calculate trip distance
data['trip_distance'] = np.sqrt((data['pickup_longitude'] - data['dropoff_longitude'])
** 2 +
(data['pickup_latitude'] - data['dropoff_latitude']) ** 2)
data['trip_duration'] = data['trip_duration'].fillna(data['trip_duration'].mean()) #
Fill NaN values

# Select relevant features
X = data[['trip_distance', 'trip_duration']].dropna().values

# Store results for each algorithm
results = {}

# Mini-Batch KMeans
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = MiniBatchKMeans(n_clusters=5, random_state=1, batch_size=512)
kmeans.fit(X_scaled)
results['MiniBatch KMeans'] = kmeans.labels_

# BIRCH
birch = Birch(n_clusters=5)
birch.fit(X_scaled)
results['BIRCH'] = birch.labels_

# CLARA (using K-Medoids for illustration)
def clara(X, n_clusters, n_local=10):
best_medoids = None
best_cost = float('inf')
for _ in range(n_local):
medoids = random.sample(range(len(X)), n_clusters)

# Calculate cost
cost = sum(np.linalg.norm(X[point] - X[medoids[i]]) for i, cluster in
enumerate(medoids) for point in cluster)
if cost < best_cost:
best_cost = cost
best_medoids = medoids
return best_medoids

# Applying CLARA
clara_labels = clara(X, 5)
results['CLARA'] = np.array(clara_labels)

# CLARANS (Randomized K-Medoids)
def clarans(X, n_clusters, n_local=10):
best_medoids = None
best_cost = float('inf')
for _ in range(n_local):
medoids = random.sample(range(len(X)), n_clusters)

# Calculate cost
cost = sum(np.linalg.norm(X[point] - X[medoids[i]]) for i, cluster in
enumerate(medoids) for point in cluster)
if cost < best_cost:
best_cost = cost
best_medoids = medoids
return best_medoids

# Apply CLARANS
clarans_labels = clarans(X, 5)
results['CLARANS'] = np.array(clarans_labels)

# CURE (Shrinking representatives)
def cure(X, n_clusters):
centroids = random.sample(list(X), n_clusters)
clusters = [[] for _ in range(n_clusters)]
for point in X:
closest_idx = np.argmin([np.linalg.norm(point - centroid) for centroid in
centroids])
clusters[closest_idx].append(point)
representatives = []
for cluster in clusters:
if len(cluster) > 0:
representatives.append(random.sample(cluster, min(5, len(cluster))))
return representatives

# Apply CURE
cure_labels = cure(X, 5)
results['CURE'] = np.array(cure_labels)

# STING (Statistical Information Grid)
def sting_grid_clustering(X, grid_size=0.01):
grid = {}
for point in X:
grid_idx = tuple(np.floor(point / grid_size))
if grid_idx not in grid:
grid[grid_idx] = []
grid[grid_idx].append(point)
return grid

# Apply STING
grid_clusters = sting_grid_clustering(X)
results['STING'] = np.array([0]*len(X)) # Placeholder for actual labels

# DENCLUE (Density-based clustering)
def denclue(X, bandwidth=0.5):
kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(X)
log_density = kde.score_samples(X)
return np.exp(log_density)

# Apply DENCLUE
density = denclue(X)
results['DENCLUE'] = np.array([0]*len(X)) # Placeholder for actual labels

# OPTICS
optics = OPTICS(min_samples=10, xi=0.1, min_cluster_size=0.1)
optics.fit(X_scaled)
results['OPTICS'] = optics.labels_

# Create a DataFrame for ARI results
algorithms = list(results.keys())
ari_table = pd.DataFrame(index=algorithms, columns=algorithms)

# Calculate ARI for each pair of algorithms
for algo1 in algorithms:
for algo2 in algorithms:
ari_score = adjusted_rand_score(results[algo1], results[algo2])
ari_table.at[algo1, algo2] = ari_score

# Convert ARI scores to float for better formatting
ari_table = ari_table.astype(float)

# Visualization
plt.figure(figsize=(10, 8))
plt.imshow(ari_table, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Adjusted Rand Index')
plt.xticks(ticks=np.arange(len(algorithms)), labels=algorithms, rotation=45)
plt.yticks(ticks=np.arange(len(algorithms)), labels=algorithms)
plt.title('Adjusted Rand Index Comparison between Clustering Algorithms')
plt.show()
# Display ARI comparison table
print(ari_table)
