import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset (stroke_processed.csv)
df = pd.read_csv('stroke_processed.csv')

# Selecting features for clustering (all except 'stroke')
X = df.drop('stroke', axis=1)

# Perform ICA with 5 components (assuming you do not scale the data)
n_components = 5
ica = FastICA(n_components=n_components, random_state=42)
X_ica = ica.fit_transform(X)

# Initialize Gaussian Mixture Model with 3 components
n_components_em = 3
gmm = GaussianMixture(n_components=n_components_em, random_state=42)

# Fit GMM on the ICA-transformed data
gmm.fit(X_ica)

# Predict clusters
cluster_labels = gmm.predict(X_ica)

# Evaluate clustering performance
ari = adjusted_rand_score(df['stroke'], cluster_labels)
silhouette = silhouette_score(X, cluster_labels)  # Use X instead of scaled X

print(f"EM Clustering Results after ICA:")
print(f"Adjusted Rand Index (ARI): {ari:.4f}")
print(f"Silhouette Score: {silhouette:.4f}")

# Add cluster labels to the original DataFrame
df['cluster_ica'] = cluster_labels

# Plotting clusters in 2D (using first two independent components)
plt.figure(figsize=(6, 6))
sns.scatterplot(x=X_ica[:, 0], y=X_ica[:, 1], hue=cluster_labels, palette='Set1', legend='full')
plt.title('EM Clustering Results after ICA')
plt.xlabel('Independent Component 1')
plt.ylabel('Independent Component 2')
plt.show()

# Analyze cluster characteristics
cluster_summary_ica = df.groupby('cluster_ica').mean()
cluster_summary_ica.to_csv('em_ica_stroke_result.csv', index=True)
cluster_sizes_ica = df['cluster_ica'].value_counts().sort_index()
print("Cluster Characteristics after ICA:")
print(cluster_summary_ica)
print("\nCluster Sizes after ICA:")
print(cluster_sizes_ica)