import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
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

# Perform ICA with 5 components
n_components = 5
ica = FastICA(n_components=n_components, random_state=42)
X_ica = ica.fit_transform(X_scaled)

# Initialize Gaussian Mixture Model with 3 components
n_components_em = 3
gmm = GaussianMixture(n_components=n_components_em, random_state=42)

# Fit GMM on the ICA-transformed data
gmm.fit(X_ica)

# Predict clusters
cluster_labels = gmm.predict(X_ica)

# Evaluate clustering performance
ari = adjusted_rand_score(df['output'], cluster_labels)
silhouette = silhouette_score(X_scaled, cluster_labels)

print(f"EM Clustering Results after ICA:")
print(f"Adjusted Rand Index (ARI): {ari:.4f}")
print(f"Silhouette Score: {silhouette:.4f}")

# Add cluster labels to the original DataFrame
df['cluster_ica'] = cluster_labels

# Plotting clusters in 2D (assuming first two components for visualization)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_ica[:, 0], y=X_ica[:, 1], hue=cluster_labels, palette='Set1', legend='full')
plt.title('EM Clustering Results after ICA')
plt.xlabel('Independent Component 1')
plt.ylabel('Independent Component 2')
plt.show()

# Analyze cluster characteristics
cluster_summary_ica = df.groupby('cluster_ica').mean()
cluster_sizes_ica = df['cluster_ica'].value_counts().sort_index()
print("Cluster Characteristics after ICA:")
print(cluster_summary_ica)
print("\nCluster Sizes after ICA:")
print(cluster_sizes_ica)