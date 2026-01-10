BIRCH

# Import required libraries
from sklearn.cluster import Birch
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Step 1: Load dataset
# Ensure the correct file path is used
data = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\trainee files\NYC.csv")

# Step 2: Feature engineering - Calculate trip distance using Euclidean formula
data['trip_distance'] = np.sqrt(
(data['pickup_longitude'] - data['dropoff_longitude'])**2 +
(data['pickup_latitude'] - data['dropoff_latitude'])**2
)

# Step 3: Data preprocessing - Select relevant features and remove any missing values
X = data[["trip_distance", "trip_duration"]].dropna()

# Step 4: Feature scaling - Normalize the data using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Apply BIRCH clustering
# BIRCH clustering with 5 clusters
birch_model = Birch(n_clusters=5)
birch_model.fit(X_scaled)

# Step 6: Predict clusters and add labels to the original dataframe
data['cluster'] = birch_model.predict(X_scaled)
print("Cluster Labels: ", data['cluster'].unique())

# Step 7: Output the subcluster centers (scaled)
print("Subcluster Centers (scaled): ", birch_model.subcluster_centers_)
# Inverse the scaling for subcluster centers to show in original scale
subcluster_centers_original =
scaler.inverse_transform(birch_model.subcluster_centers_)
print("Subcluster Centers (original scale): ", subcluster_centers_original)

# Step 8: Visualization - Scatter plot of clusters with data points
# Create a DataFrame with scaled features and predicted cluster labels
X_scaled_df = pd.DataFrame(X_scaled, columns=["trip_distance", "trip_duration"])
X_scaled_df['cluster'] = data['cluster']
# Plot the clusters based on scaled trip distance and trip duration
plt.figure(figsize=(10, 6))
palette = sns.color_palette('Set2', n_colors=5)
sns.scatterplot(x='trip_distance', y='trip_duration', hue='cluster', data=X_scaled_df,
palette=palette)
plt.title("BIRCH Clustering (Scaled Features)")
plt.xlabel("Scaled Trip Distance")
plt.ylabel("Scaled Trip Duration")
plt.legend(loc="upper right")
plt.show()

# Step 9: Visualization - Show subcluster centers on original scale
plt.figure(figsize=(10, 6))
plt.scatter(data['trip_distance'], data['trip_duration'], c=data['cluster'],
cmap='viridis', marker='o', alpha=0.6, label='Data points')
plt.scatter(subcluster_centers_original[:, 0], subcluster_centers_original[:, 1], c='red',
marker='X', s=200, label='Subcluster Centers')
plt.title("Subcluster Centers on Original Data")
plt.xlabel("Trip Distance")
plt.ylabel("Trip Duration")
plt.legend(loc="upper right")
plt.show()