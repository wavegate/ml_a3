import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv('stroke_processed.csv')

# Selecting features for clustering (all except 'stroke')
X = df.drop('stroke', axis=1)

# Define range of n_components to test
n_components_range = range(2, 11)  # Example range from 2 to 10

# Initialize lists to store results
ari_scores = []
silhouette_scores = []

# Iterate over different n_components
for n_components in n_components_range:
    # Initialize Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    
    # Fit GMM
    gmm.fit(X)
    
    # Predict clusters
    cluster_labels = gmm.predict(X)
    
    # Calculate ARI
    ari = adjusted_rand_score(df['stroke'], cluster_labels)
    ari_scores.append(ari)
    
    # Calculate Silhouette Score
    silhouette = silhouette_score(X, cluster_labels)
    silhouette_scores.append(silhouette)
    
    print(f"n_components={n_components}: ARI={ari:.4f}, Silhouette={silhouette:.4f}")

# Plotting results (optional)
plt.figure(figsize=(6, 6))
plt.plot(n_components_range, ari_scores, marker='o', linestyle='-', color='b', label='ARI')
plt.plot(n_components_range, silhouette_scores, marker='o', linestyle='-', color='r', label='Silhouette')
plt.title('ARI and Silhouette Score vs n_components')
plt.xlabel('n_components')
plt.ylabel('Score')
plt.xticks(n_components_range)
plt.legend()
plt.grid(True)
plt.show()

# Choosing n_components = 4 for further analysis (you can adjust this based on the plotted results)
chosen_n_components = 4

# Initialize Gaussian Mixture Model with chosen_n_components
gmm = GaussianMixture(n_components=chosen_n_components, random_state=42)
gmm.fit(X)
cluster_labels = gmm.predict(X)

# Add cluster labels to the original DataFrame
df['cluster'] = cluster_labels

# Print cluster means
print("Cluster Means:")
print(gmm.means_)

# Analyze cluster characteristics
cluster_summary = df.groupby('cluster').mean()
cluster_sizes = df['cluster'].value_counts().sort_index()

# Save cluster summary to CSV
cluster_summary.to_csv('em_stroke_result.csv', index=True)

print("Cluster Characteristics:")
print(cluster_summary)
print("\nCluster Sizes:")
print(cluster_sizes)