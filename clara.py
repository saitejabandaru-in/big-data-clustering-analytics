CLARA
from pyclustering.cluster.clara import clara
import pandas as pd
import numpy as np
from pyclustering.utils import read_sample
from pyclustering.cluster import cluster_visualizer

# Load dataset
data = pd.read_csv(r"trainee files\NYC.csv")

# Feature engineering: calculate trip distance using coordinates
data['trip_distance'] = np.sqrt((data['pickup_longitude'] -
data['dropoff_longitude'])**2 +
(data['pickup_latitude'] - data['dropoff_latitude'])**2)

# Select features and remove missing values
X = data[['trip_distance', 'trip_duration']].dropna().values

# Apply CLARA with 5 clusters
clara_instance = clara(X, 5)
clara_instance.process()

# Get clusters and medoids
clusters = clara_instance.get_clusters()
medoids = clara_instance.get_medoids()

# Visualization
visualizer = cluster_visualizer()
visualizer.append_clusters(clusters, X)
visualizer.show()
from sklearn.metrics import silhouette_score
labels = np.zeros(len(X))
for i, cluster in enumerate(clusters):
labels[cluster] = i
silhouette_avg = silhouette_score(X, labels)
print(f"Silhouette Score: {silhouette_avg}")
