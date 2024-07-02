import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


# 1. age - age in years

# 2. sex - sex (1 = male; 0 = female)

# 3. cp - chest pain type (1 = typical angina; 2 = atypical angina; 3 = non-anginal pain; 0 = asymptomatic)

# 4. trestbps - resting blood pressure (in mm Hg on admission to the hospital)

# 5. chol - serum cholestoral in mg/dl

# 6. fbs - fasting blood sugar > 120 mg/dl (1 = true; 0 = false)

# 7. restecg - resting electrocardiographic results (1 = normal; 2 = having ST-T wave abnormality; 0 = hypertrophy)

# 8. thalach - maximum heart rate achieved

# 9. exang - exercise induced angina (1 = yes; 0 = no)

# 10. oldpeak - ST depression induced by exercise relative to rest

# 11. slope - the slope of the peak exercise ST segment (2 = upsloping; 1 = flat; 0 = downsloping)

# 12. ca - number of major vessels (0-3) colored by flourosopy

# 13. thal - 2 = normal; 1 = fixed defect; 3 = reversable defect

# 14. num - the predicted attribute - diagnosis of heart disease (angiographic disease status) (Value 0 = < diameter narrowing; Value 1 = > 50% diameter narrowing)

# Load your dataset
df = pd.read_csv('heart.csv')

# Selecting features for clustering (all except 'output')
X = df.drop('output', axis=1)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

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
    gmm.fit(X_scaled)
    
    # Predict clusters
    cluster_labels = gmm.predict(X_scaled)
    
    # Calculate ARI
    ari = adjusted_rand_score(df['output'], cluster_labels)
    ari_scores.append(ari)
    
    # Calculate Silhouette Score
    silhouette = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(silhouette)
    
    print(f"n_components={n_components}: ARI={ari:.4f}, Silhouette={silhouette:.4f}")

# Plotting results (optional)
plt.figure(figsize=(6, 6))
plt.plot(n_components_range, ari_scores, marker='o', linestyle='-', color='b', label='ARI')
plt.plot(n_components_range, silhouette_scores, marker='o', linestyle='-', color='r', label='Silhouette')
plt.title('ARI/Silhouette for Heart dataset')
plt.xlabel('n_components')
plt.ylabel('Score')
plt.xticks(n_components_range)
plt.legend()
plt.grid(True)
plt.show()

# Choosing n_components = 3 for further analysis
chosen_n_components = 3

# Initialize Gaussian Mixture Model with chosen_n_components
gmm = GaussianMixture(n_components=chosen_n_components, random_state=42)
gmm.fit(X_scaled)
cluster_labels = gmm.predict(X_scaled)

# Add cluster labels to the original DataFrame
df['cluster'] = cluster_labels

# Print cluster means
print("Cluster Means:")
print(gmm.means_)

# Analyze cluster characteristics
cluster_summary = df.groupby('cluster').mean()
cluster_sizes = df['cluster'].value_counts().sort_index()
print("Cluster Characteristics:")
print(cluster_summary)
print("\nCluster Sizes:")
print(cluster_sizes)