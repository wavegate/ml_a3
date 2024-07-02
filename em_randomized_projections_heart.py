import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection
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

# Perform Random Projections with 5 components
n_components = 5
grp = GaussianRandomProjection(n_components=n_components, random_state=42)
X_projected = grp.fit_transform(X_scaled)

# Initialize Gaussian Mixture Model with 3 components
n_components_em = 3
gmm = GaussianMixture(n_components=n_components_em, random_state=42)

# Fit GMM on the projected data
gmm.fit(X_projected)

# Predict clusters
cluster_labels = gmm.predict(X_projected)

# Evaluate clustering performance
ari = adjusted_rand_score(df['output'], cluster_labels)
silhouette = silhouette_score(X_scaled, cluster_labels)

print(f"EM Clustering Results after Random Projections:")
print(f"Adjusted Rand Index (ARI): {ari:.4f}")
print(f"Silhouette Score: {silhouette:.4f}")

# Add cluster labels to the original DataFrame
df['cluster_rp'] = cluster_labels

# Plotting clusters in 2D (assuming first two components for visualization)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_projected[:, 0], y=X_projected[:, 1], hue=cluster_labels, palette='Set1', legend='full')
plt.title('EM Clustering Results after Random Projections')
plt.xlabel('Projected Feature 1')
plt.ylabel('Projected Feature 2')
plt.show()

# Analyze cluster characteristics
cluster_summary_rp = df.groupby('cluster_rp').mean()
cluster_sizes_rp = df['cluster_rp'].value_counts().sort_index()
print("Cluster Characteristics after Random Projections:")
print(cluster_summary_rp)
print("\nCluster Sizes after Random Projections:")
print(cluster_sizes_rp)