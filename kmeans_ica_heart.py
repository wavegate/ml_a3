import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv('heart.csv')

# Selecting features for clustering (all except 'output')
X = df.drop('output', axis=1)

# Optionally, scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform ICA with 2 components for visualization (optional)
n_components = 2
ica = FastICA(n_components=n_components, random_state=42)
X_ica = ica.fit_transform(X_scaled)

# Initialize K-means clustering with 2 clusters
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X_ica)

# Evaluate clustering performance
ari = adjusted_rand_score(df['output'], cluster_labels)
silhouette = silhouette_score(X_scaled, cluster_labels)

print(f"K-means Clustering Results with {n_clusters} clusters after ICA:")
print(f"Adjusted Rand Index (ARI): {ari:.4f}")
print(f"Silhouette Score: {silhouette:.4f}")

# Add cluster labels to the original DataFrame
df['cluster_ica_kmeans'] = cluster_labels

# Plotting clusters in 2D space (using ICA components for visualization)
plt.figure(figsize=(6, 6))
sns.scatterplot(x=X_ica[:, 0], y=X_ica[:, 1], hue=cluster_labels, palette='Set1', legend='full')
plt.title(f'K-means Clustering Results with {n_clusters} clusters after ICA')
plt.xlabel('Independent Component 1')
plt.ylabel('Independent Component 2')
plt.show()

# Analyze cluster characteristics
cluster_summary = df.groupby('cluster_ica_kmeans').mean()
cluster_sizes = df['cluster_ica_kmeans'].value_counts().sort_index()
print("Cluster Characteristics after K-means Clustering with ICA:")
print(cluster_summary)
print("\nCluster Sizes after K-means Clustering with ICA:")
print(cluster_sizes)