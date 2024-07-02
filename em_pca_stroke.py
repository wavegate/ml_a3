import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset (stroke_processed.csv)
df = pd.read_csv('stroke_processed.csv')

# Selecting features for clustering (all except 'stroke')
X = df.drop('stroke', axis=1)

# Perform PCA with 5 components (assuming you do not scale the data)
n_components = 5
pca = PCA(n_components=n_components, random_state=42)
X_pca = pca.fit_transform(X)

# Initialize Gaussian Mixture Model with 3 components
n_components_em = 3
gmm = GaussianMixture(n_components=n_components_em, random_state=42)

# Fit GMM on the PCA-transformed data
gmm.fit(X_pca)

# Predict clusters
cluster_labels = gmm.predict(X_pca)

# Evaluate clustering performance
ari = adjusted_rand_score(df['stroke'], cluster_labels)
silhouette = silhouette_score(X, cluster_labels)  # Use X instead of scaled X

print(f"EM Clustering Results after PCA:")
print(f"Adjusted Rand Index (ARI): {ari:.4f}")
print(f"Silhouette Score: {silhouette:.4f}")

# Add cluster labels to the original DataFrame
df['cluster_pca'] = cluster_labels

# Plotting clusters in 2D (using first two principal components)
plt.figure(figsize=(6, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cluster_labels, palette='Set1', legend='full')
plt.title('EM Clustering Results after PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Analyze cluster characteristics
cluster_summary_pca = df.groupby('cluster_pca').mean()
cluster_summary_pca.to_csv('em_pca_stroke_result.csv', index=True)
cluster_sizes_pca = df['cluster_pca'].value_counts().sort_index()
print("Cluster Characteristics after PCA:")
print(cluster_summary_pca)
print("\nCluster Sizes after PCA:")
print(cluster_sizes_pca)