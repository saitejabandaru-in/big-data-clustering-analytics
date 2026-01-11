Mini-Batch Kmeans
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set environment variable to prevent MKL-related memory leak in Windows
os.environ["OMP_NUM_THREADS"] = "1"

# Step 1: Load the dataset
df = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\trainee files\NYC.csv")

# Step 2: Feature Engineering - Calculate Trip Distance
# Compute the Euclidean distance based on pickup and dropoff coordinates
df['trip_distance'] = np.sqrt(
(df['pickup_longitude'] - df['dropoff_longitude'])**2 +
(df['pickup_latitude'] - df['dropoff_latitude'])**2
)

# Step 3: Data Preprocessing - Scale the features (trip_distance, trip_duration)
scaler = StandardScaler()
X = scaler.fit_transform(df[['trip_distance', 'trip_duration']])

# Step 4: Apply Mini-Batch K-Means Clustering
# Use a larger batch_size to avoid the memory leak and improve performance
kmeans = MiniBatchKMeans(n_clusters=5, random_state=1, batch_size=512)
kmeans.fit(X)

# Step 5: Output the cluster centers in scaled form
print("Cluster Centers (scaled):", kmeans.cluster_centers_)
# Assign the predicted clusters to the original DataFrame
df['cluster'] = kmeans.predict(X)

# Step 6: Inverse transform the cluster centers to original scale (before scaling)
cluster_centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
print("Cluster Centers (original scale):", cluster_centers_original)

# Step 7: Visualization - Scatter plot of clusters with data points

# Prepare a DataFrame for the scaled features with clusters
df_scaled = pd.DataFrame(X, columns=['trip_distance', 'trip_duration'])
df_scaled['cluster'] = df['cluster']

# Create a color palette for the clusters
palette = sns.color_palette('Set1', n_colors=5)

# Scatter plot for visualizing clusters based on scaled trip_distance and trip_duration
plt.figure(figsize=(10, 6))
sns.scatterplot(x='trip_distance', y='trip_duration', hue='cluster', data=df_scaled,
palette=palette)
plt.title("Mini-Batch K-Means Clustering (Scaled Features)")
plt.xlabel("Scaled Trip Distance")
plt.ylabel("Scaled Trip Duration")
plt.legend(loc="upper right") # Avoiding 'best' for speed
plt.show()

# Step 8: Visualization - Cluster centers on the original scale
plt.figure(figsize=(10, 6))
plt.scatter(df['trip_distance'], df['trip_duration'], c=df['cluster'], cmap='viridis',
marker='o', alpha=0.6, label='Data points')
plt.scatter(cluster_centers_original[:, 0], cluster_centers_original[:, 1], c='red',
marker='X', s=200, label='Cluster Centers')
plt.title("Cluster Centers on Original Data")
plt.xlabel("Trip Distance")
plt.ylabel("Trip Duration")
plt.legend(loc="upper right") # Avoid using 'best' to speed up processing
plt.show()
