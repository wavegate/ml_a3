import pandas as pd
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv('stroke_processed.csv')

# Selecting features for clustering (all except 'stroke')
X = df.drop('stroke', axis=1)

# Perform Random Projections with 2 components for visualization (optional)
n_components = 5
grp = GaussianRandomProjection(n_components=n_components, random_state=42)
X_grp = grp.fit_transform(X)

# Initialize K-means clustering with 4 clusters
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X_grp)

# Evaluate clustering performance
ari = adjusted_rand_score(df['stroke'], cluster_labels)
silhouette = silhouette_score(X, cluster_labels)

print(f"K-means Clustering Results with {n_clusters} clusters after Randomized Projections:")
print(f"Adjusted Rand Index (ARI): {ari:.4f}")
print(f"Silhouette Score: {silhouette:.4f}")

# Add cluster labels to the original DataFrame
df['cluster_rp_kmeans'] = cluster_labels

# Plotting clusters in 2D space (using Randomized Projections for visualization)
plt.figure(figsize=(6, 6))
sns.scatterplot(x=X_grp[:, 0], y=X_grp[:, 1], hue=cluster_labels, palette='Set1', legend='full')
plt.title(f'K-means Clustering Results with {n_clusters} clusters after Randomized Projections')
plt.xlabel('Projected Feature 1')
plt.ylabel('Projected Feature 2')
plt.show()

# Analyze cluster characteristics
cluster_summary = df.groupby('cluster_rp_kmeans').mean()
cluster_summary.to_csv('kmeans_randomized_projections_stroke_result.csv', index=True)
cluster_sizes = df['cluster_rp_kmeans'].value_counts().sort_index()
print("Cluster Characteristics after K-means Clustering with Randomized Projections:")
print(cluster_summary)
print("\nCluster Sizes after K-means Clustering with Randomized Projections:")
print(cluster_sizes)