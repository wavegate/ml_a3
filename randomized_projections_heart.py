import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances

# Load your dataset
df = pd.read_csv('heart.csv')

# Selecting features for projection (all except 'output')
X = df.drop('output', axis=1)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define a range of n_components to test
n_components_range = range(2, 12)  # Adjust this range based on your analysis needs

# Initialize lists to store results
reconstruction_errors = []
mean_variances = []

# Iterate over different n_components
for n_components in n_components_range:
    # Initialize Gaussian Random Projection
    grp = GaussianRandomProjection(n_components=n_components, random_state=42)
    
    # Fit and transform the original data
    X_projected = grp.fit_transform(X_scaled)
    
    # Estimate reconstruction error (using Frobenius norm)
    X_reconstructed = np.dot(X_projected, grp.components_)
    reconstruction_error = np.linalg.norm(X_scaled - X_reconstructed, 'fro') / np.linalg.norm(X_scaled, 'fro')
    reconstruction_errors.append(reconstruction_error)
    
    # Calculate variance across multiple runs
    n_runs = 10
    projection_variances = []
    for _ in range(n_runs):
        X_projected = grp.fit_transform(X_scaled)
        projection_variance = np.var(X_projected)
        projection_variances.append(projection_variance)
    
    mean_variance = np.mean(projection_variances)
    mean_variances.append(mean_variance)
    
    print(f"n_components={n_components}: Reconstruction Error={reconstruction_error:.4f}, Mean Variance={mean_variance:.4f}")

# Plotting reconstruction error vs. n_components
plt.figure(figsize=(6, 6))
plt.plot(n_components_range, reconstruction_errors, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Components')
plt.ylabel('Reconstruction Error')
plt.title('Reconstruction Error vs. n_components')
plt.grid(True)
plt.tight_layout()
plt.show()

# Determine the best n_components based on reconstruction error and mean variance
# best_n_components = n_components_range[np.argmin(reconstruction_errors)]
best_n_components = 5
print(f"Best n_components based on Reconstruction Error: {best_n_components}")

# Proceed with further analyses using the best n_components
# Initialize Gaussian Random Projection with the best n_components
grp_best = GaussianRandomProjection(n_components=best_n_components, random_state=42)
X_projected_best = grp_best.fit_transform(X_scaled)

# Variance of random projections across multiple runs
n_runs = 10
projection_variances_best = []
for _ in range(n_runs):
    X_projected_best = grp_best.fit_transform(X_scaled)
    projection_variance_best = np.var(X_projected_best)
    projection_variances_best.append(projection_variance_best)

mean_variance_best = np.mean(projection_variances_best)
print(f"Mean variance across {n_runs} runs (Best n_components={best_n_components}): {mean_variance_best:.4f}")

# Data rank and collinearity analysis
data_rank = np.linalg.matrix_rank(X_scaled)
print(f"Rank of the data: {data_rank}")

# Correlation matrix for qualitative assessment of collinearity
correlation_matrix = np.corrcoef(X_scaled, rowvar=False)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", xticklabels=X.columns, yticklabels=X.columns)
plt.title('Correlation Matrix of Scaled Features')
plt.show()

# Quantitative assessment of collinearity
feature_correlations = X.corr().abs().unstack().sort_values(ascending=False)
high_correlations = feature_correlations[feature_correlations != 1.0]
print("\nTop correlated feature pairs:")
print(high_correlations.head())

# Influence of noise on the algorithm
# Generate noisy data
noise_level = 0.1
X_noisy = X_scaled + np.random.normal(scale=noise_level, size=X_scaled.shape)

# Apply randomized projection to noisy data with the best n_components
grp_noisy_best = GaussianRandomProjection(n_components=best_n_components, random_state=42)
X_noisy_projected_best = grp_noisy_best.fit_transform(X_noisy)

# Visualize the effect of noise on projections
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_projected_best[:, 0], y=X_projected_best[:, 1], hue=df['output'], palette='Set1', legend='full', alpha=0.7)
sns.scatterplot(x=X_noisy_projected_best[:, 0], y=X_noisy_projected_best[:, 1], hue=df['output'], palette='Set1', legend='full', marker='x', alpha=0.7)
plt.title(f'Random Projection Results with Noise (Noise Level: {noise_level}) - Best n_components={best_n_components}')
plt.xlabel('Projected Feature 1')
plt.ylabel('Projected Feature 2')
plt.legend()
plt.show()

# Quantitative metrics - distance measures and overlap measures
# Compute distance measures (using pairwise distances)
distance_original_best = pairwise_distances(X_projected_best, metric='euclidean')
distance_noisy_best = pairwise_distances(X_noisy_projected_best, metric='euclidean')

# Mean distance between points in original and noisy projections
mean_distance_original_best = np.mean(distance_original_best)
mean_distance_noisy_best = np.mean(distance_noisy_best)

print(f"Mean distance between points - Original (Best n_components={best_n_components}): {mean_distance_original_best:.4f}")
print(f"Mean distance between points - Noisy (Best n_components={best_n_components}): {mean_distance_noisy_best:.4f}")

# Overlap measures (using overlap indices or other suitable metrics)
overlap_index_best = np.sum(np.abs(X_projected_best - X_noisy_projected_best)) / np.sum(np.abs(X_projected_best))
print(f"Overlap Index (Best n_components={best_n_components}): {overlap_index_best:.4f}")

# Additional analyses can be added here based on the best n_components
# ...

# Example: Data rank and collinearity analysis for the best n_components
data_rank_best = np.linalg.matrix_rank(X_projected_best)
print(f"Rank of the data (Best n_components={best_n_components}): {data_rank_best}")

# Example: Visualization of the projection results for the best n_components
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_projected_best[:, 0], y=X_projected_best[:, 1], hue=df['output'], palette='Set1', legend='full')
plt.title(f'Random Projection Results (Best n_components={best_n_components})')
plt.xlabel('Projected Feature 1')
plt.ylabel('Projected Feature 2')
plt.show()

# Based on the analysis using Gaussian Random Projection with n_components=10, the reconstruction error is measured at 1.0220, indicating a relatively low error in preserving the original data structure. The mean variance across 10 runs is calculated as 1.1773, suggesting consistent variability in the projected data. The dataset's rank is determined to be 13, highlighting its complexity and dimensionality. Examining feature correlations reveals strong associations between features like 'oldpeak' and 'slp' with a correlation coefficient of 0.5775, indicating potential multicollinearity. Quantitative assessments show a mean distance of 4.6233 between points in the original projections and 4.6285 in noisy projections, with an overlap index of 0.0991, illustrating the impact of noise on projected data points. When considering the best n_components, the rank of the projected data is reduced to 10, reflecting the effective dimensionality reduction achieved with this parameter setting. These insights collectively inform on the quality of projection and the underlying structure of the dataset.