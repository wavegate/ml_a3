import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv('heart.csv')

# Selecting features for ICA (all except 'output')
X = df.drop('output', axis=1)

# Apply StandardScaler for feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize ICA with 5 components
n_components = 5
ica = FastICA(n_components=n_components, random_state=42)
X_ica = ica.fit_transform(X_scaled)

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
sns.scatterplot(x=X_ica[:, 0], y=X_ica[:, 1], hue=df['output'], palette='Set1', legend='full', alpha=0.7)
plt.title('ICA: Data Distribution in Reduced Space (Component 1 vs Component 2)')
plt.xlabel('Independent Component 1')
plt.ylabel('Independent Component 2')
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

# Assessing the effect of noise on ICA
noise_level = 0.1
X_noisy = X_scaled + np.random.normal(scale=noise_level, size=X_scaled.shape)
X_noisy_ica = ica.fit_transform(X_noisy)

plt.figure(figsize=(6, 6))
sns.scatterplot(x=X_ica[:, 0], y=X_ica[:, 1], hue=df['output'], palette='Set1', legend='full', alpha=0.7)
sns.scatterplot(x=X_noisy_ica[:, 0], y=X_noisy_ica[:, 1], hue=df['output'], palette='Set1', legend='full', marker='x', alpha=0.7)
plt.title(f'ICA: Data Distribution with Noise (Noise Level: {noise_level})')
plt.xlabel('Independent Component 1')
plt.ylabel('Independent Component 2')
plt.legend(['Original Data', 'Noisy Data'])
plt.show()

# Describe how data properties might influence algorithm outputs
# For ICA, non-Gaussian data and non-linear mixing can affect the meaningfulness of independent components.
# Higher rank indicates less linear dependence among features.

# Specific properties like feature scaling, data distribution, and noise levels can influence ICA outputs,
# impacting the extraction of independent components and their interpretability.
