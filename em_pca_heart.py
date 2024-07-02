import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
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

# Perform PCA with 5 components
n_components = 5
pca = PCA(n_components=n_components, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Initialize Gaussian Mixture Model with 3 components
n_components_em = 3
gmm = GaussianMixture(n_components=n_components_em, random_state=42)

# Fit GMM on the PCA-transformed data
gmm.fit(X_pca)

# Predict clusters
cluster_labels = gmm.predict(X_pca)

# Evaluate clustering performance
ari = adjusted_rand_score(df['output'], cluster_labels)
silhouette = silhouette_score(X_scaled, cluster_labels)

print(f"EM Clustering Results after PCA:")
print(f"Adjusted Rand Index (ARI): {ari:.4f}")
print(f"Silhouette Score: {silhouette:.4f}")

# Add cluster labels to the original DataFrame
df['cluster_pca'] = cluster_labels

# Plotting clusters in 2D (assuming first two components for visualization)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cluster_labels, palette='Set1', legend='full')
plt.title('EM Clustering Results after PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Analyze cluster characteristics
cluster_summary_pca = df.groupby('cluster_pca').mean()
cluster_sizes_pca = df['cluster_pca'].value_counts().sort_index()
print("Cluster Characteristics after PCA:")
print(cluster_summary_pca)
print("\nCluster Sizes after PCA:")
print(cluster_sizes_pca)