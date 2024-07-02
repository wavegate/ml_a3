import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
from sklearn.metrics import pairwise_distances
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv('stroke_processed.csv')

# Selecting features for ICA (all except 'stroke')
X = df.drop('stroke', axis=1)

# Initialize ICA with 5 components
n_components = 5
ica = FastICA(n_components=n_components, random_state=42)
X_ica = ica.fit_transform(X)

# Plotting the kurtosis of independent components
kurtosis = np.mean(np.abs(ica.components_), axis=1)
print(kurtosis)
plt.figure(figsize=(6, 6))
plt.bar(range(1, n_components + 1), kurtosis, alpha=0.8, align='center')
plt.xlabel('Independent Components')
plt.ylabel('Kurtosis')
plt.title('Kurtosis of Independent Components for ICA')
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualize the data in ICA space
plt.figure(figsize=(6, 6))
sns.scatterplot(x=X_ica[:, 0], y=X_ica[:, 1], hue=df['stroke'], palette='Set1', legend='full', alpha=0.7)
plt.title('ICA: Data Distribution in Reduced Space (IC1 vs IC2)')
plt.xlabel('Independent Component 1')
plt.ylabel('Independent Component 2')
plt.legend()
plt.show()

# Assessing the rank of the data
data_rank = np.linalg.matrix_rank(X)
print(f"Rank of the original data: {data_rank}")

# Assessing collinearity qualitatively and quantitatively
correlation_matrix = np.corrcoef(X, rowvar=False)
plt.figure(figsize=(6, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", xticklabels=X.columns, yticklabels=X.columns)
plt.title('Correlation Matrix of Scaled Features')
plt.show()

feature_correlations = X.corr().abs().unstack().sort_values(ascending=False)
high_correlations = feature_correlations[feature_correlations != 1.0]
print("\nTop correlated feature pairs:")
print(high_correlations.head())

# Assessing the effect of noise on ICA
noise_level = 0.1
X_noisy = X + np.random.normal(scale=noise_level, size=X.shape)
X_noisy_ica = ica.fit_transform(X_noisy)

plt.figure(figsize=(6, 6))
sns.scatterplot(x=X_ica[:, 0], y=X_ica[:, 1], hue=df['stroke'], palette='Set1', legend='full', alpha=0.7)
sns.scatterplot(x=X_noisy_ica[:, 0], y=X_noisy_ica[:, 1], hue=df['stroke'], palette='Set1', legend='full', marker='x', alpha=0.7)
plt.title(f'ICA: Data Distribution with Noise (Noise Level: {noise_level})')
plt.xlabel('Independent Component 1')
plt.ylabel('Independent Component 2')
plt.legend(['Original Data', 'Noisy Data'])
plt.show()

# Quantitative metrics - distance measures and overlap measures
# Compute distance measures (using pairwise distances)
distance_original = pairwise_distances(X_ica, metric='euclidean')
distance_noisy = pairwise_distances(X_noisy_ica, metric='euclidean')

# Mean distance between points in original and noisy projections
mean_distance_original = np.mean(distance_original)
mean_distance_noisy = np.mean(distance_noisy)

print(f"Mean distance between points - Original: {mean_distance_original:.4f}")
print(f"Mean distance between points - Noisy: {mean_distance_noisy:.4f}")

# Overlap measures (using overlap indices or other suitable metrics)
overlap_index = np.sum(np.abs(X_ica - X_noisy_ica)) / np.sum(np.abs(X_ica))
print(f"Overlap Index: {overlap_index:.4f}")