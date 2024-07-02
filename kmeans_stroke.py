import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv('stroke_processed.csv')

# Selecting features for clustering (all except 'stroke')
X = df.drop('stroke', axis=1)

# Define range of n_clusters to test
n_clusters_range = range(2, 11)  # Example range from 2 to 10

# Initialize lists to store results
ari_scores = []
silhouette_scores = []

# Iterate over different n_clusters
for n_clusters in n_clusters_range:
    # Initialize KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    
    # Fit KMeans
    kmeans.fit(X)
    
    # Predict clusters
    cluster_labels = kmeans.predict(X)
    
    # Calculate ARI
    ari = adjusted_rand_score(df['stroke'], cluster_labels)
    ari_scores.append(ari)
    
    # Calculate Silhouette Score
    silhouette = silhouette_score(X, cluster_labels)
    silhouette_scores.append(silhouette)
    
    print(f"n_clusters={n_clusters}: ARI={ari:.4f}, Silhouette={silhouette:.4f}")

# Plotting results (optional)
plt.figure(figsize=(6, 6))
plt.plot(n_clusters_range, ari_scores, marker='o', linestyle='-', color='b', label='ARI')
plt.plot(n_clusters_range, silhouette_scores, marker='o', linestyle='-', color='r', label='Silhouette')
plt.title('ARI and Silhouette Score vs n_clusters')
plt.xlabel('n_clusters')
plt.ylabel('Score')
plt.xticks(n_clusters_range)
plt.legend()
plt.grid(True)
plt.show()

# Choosing n_clusters = 4 for further analysis (you can adjust this based on the plotted results)
chosen_n_clusters = 4

# Initialize KMeans with chosen_n_clusters
kmeans = KMeans(n_clusters=chosen_n_clusters, random_state=42)
kmeans.fit(X)
cluster_labels = kmeans.predict(X)

# Add cluster labels to the original DataFrame
df['cluster'] = cluster_labels

# Analyze cluster characteristics
cluster_summary = df.groupby('cluster').mean()
cluster_sizes = df['cluster'].value_counts().sort_index()

# Save cluster summary to CSV
cluster_summary.to_csv('kmeans_stroke_result.csv', index=True)

print("Cluster Characteristics:")
print(cluster_summary)
print("\nCluster Sizes:")
print(cluster_sizes)