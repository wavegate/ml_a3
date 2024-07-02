import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv('heart.csv')

# Selecting features for PCA (all except 'output')
X = df.drop('output', axis=1)

# Apply StandardScaler for feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize PCA with 5 components
n_components = 5
pca = PCA(n_components=n_components, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Convert X_pca to a DataFrame
X_pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])

# Save the PCA-transformed data to a CSV file
X_pca_df.to_csv('X_pca.csv', index=False)

# Plotting the distribution of eigenvalues (explained variance)
eigenvalues = pca.explained_variance_
print(eigenvalues)
plt.figure(figsize=(6, 6))
plt.bar(range(1, n_components + 1), eigenvalues, alpha=0.8, align='center')
plt.xlabel('Principal Components')
plt.ylabel('Eigenvalues')
plt.title('Distribution of Eigenvalues (Explained Variance) for PCA')
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualize the data in PCA space
plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['output'], palette='Set1', legend='full', alpha=0.7)
plt.title('PCA: Data Distribution in Reduced Space (PC1 vs PC2)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

# Assessing the rank of the data
data_rank = np.linalg.matrix_rank(X_scaled)
print(f"Rank of the original data: {data_rank}")

# Assessing collinearity qualitatively and quantitatively
correlation_matrix = np.corrcoef(X_scaled, rowvar=False)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", xticklabels=X.columns, yticklabels=X.columns)
plt.title('Correlation Matrix of Scaled Features')
plt.show()

feature_correlations = X.corr().abs().unstack().sort_values(ascending=False)
high_correlations = feature_correlations[feature_correlations != 1.0]
print("\nTop correlated feature pairs:")
print(high_correlations.head())

# Assessing the effect of noise on PCA
noise_level = 0.1
X_noisy = X_scaled + np.random.normal(scale=noise_level, size=X_scaled.shape)
X_noisy_pca = pca.fit_transform(X_noisy)

plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['output'], palette='Set1', legend='full', alpha=0.7)
sns.scatterplot(x=X_noisy_pca[:, 0], y=X_noisy_pca[:, 1], hue=df['output'], palette='Set1', legend='full', marker='x', alpha=0.7)
plt.title(f'PCA: Data Distribution with Noise (Noise Level: {noise_level})')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(['Original Data', 'Noisy Data'])
plt.show()

# Describe how data properties might influence algorithm outputs
# For PCA, data with high collinearity might lead to fewer informative principal components.
# Noise can distort principal components and reduce their explanatory power.
# Higher rank indicates less linear dependence among features.

# Specific properties like feature scaling, data distribution, and noise levels can affect PCA outcomes,
# influencing how variance is captured and represented in reduced dimensions.