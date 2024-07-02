import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv('heart.csv')

# Selecting features for clustering (all except 'output')
X = df.drop('output', axis=1)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define range of n_clusters to test
n_clusters_range = range(2, 11)  # Example range from 2 to 10

# Initialize lists to store results
ari_scores = []
silhouette_scores = []

# Iterate over different n_clusters
for n_clusters in n_clusters_range:
    # Initialize KMeans model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    
    # Fit KMeans
    kmeans.fit(X_scaled)
    
    # Predict clusters
    cluster_labels = kmeans.labels_
    
    # Calculate ARI
    ari = adjusted_rand_score(df['output'], cluster_labels)
    ari_scores.append(ari)
    
    # Calculate Silhouette Score
    silhouette = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(silhouette)
    
    print(f"n_clusters={n_clusters}: ARI={ari:.4f}, Silhouette={silhouette:.4f}")

# Plotting results (optional)
plt.figure(figsize=(6, 6))
plt.plot(n_clusters_range, ari_scores, marker='o', linestyle='-', color='b', label='ARI')
plt.plot(n_clusters_range, silhouette_scores, marker='o', linestyle='-', color='r', label='Silhouette')
plt.title('ARI and Silhouette Score vs n_clusters (KMeans)')
plt.xlabel('n_clusters')
plt.ylabel('Score')
plt.xticks(n_clusters_range)
plt.legend()
plt.grid(True)
plt.show()

# Choosing n_clusters = 2 for further analysis
chosen_n_clusters = 2

# Initialize KMeans with chosen_n_clusters
kmeans = KMeans(n_clusters=chosen_n_clusters, random_state=42)
kmeans.fit(X_scaled)
cluster_labels = kmeans.labels_

# Add cluster labels to the original DataFrame
df['cluster'] = cluster_labels

# Print cluster centers (centroids)
print("Cluster Centers (Centroids):")
print(kmeans.cluster_centers_)

# Plotting clusters in 2D (assuming first two features for visualization)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=cluster_labels, palette='Set1', legend='full')
plt.title('Clustering Results (KMeans)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Analyze cluster characteristics
cluster_summary = df.groupby('cluster').mean()
cluster_sizes = df['cluster'].value_counts().sort_index()
print("Cluster Characteristics:")
print(cluster_summary)
print("\nCluster Sizes:")
print(cluster_sizes)

# Cluster 0 tends to have slightly younger individuals with higher heart disease risk factors such as higher cholesterol levels, higher heart rates, and more pronounced ST segment patterns during exercise. Cluster 1, on the other hand, includes older individuals with lower overall risk factors for heart disease, including lower cholesterol levels and lower incidence of significant ST segment abnormalities during exercise.