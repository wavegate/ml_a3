import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# Load your dataset
df = pd.read_csv('heart.csv')

# Split data into features (X) and target (y)
X = df.drop('output', axis=1)
y = df['output']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 1: Apply EM clustering
n_components = 3  # Number of clusters
em = GaussianMixture(n_components=n_components, random_state=42)
cluster_labels_em = em.fit_predict(X)

# Append EM cluster labels as new feature to the dataset
df_em = df.copy()
df_em['cluster_em'] = cluster_labels_em

# Split data into features with EM clusters and target
X_with_em_clusters = df_em.drop('output', axis=1)
y_em = df_em['output']

# Split into training and testing sets with EM clusters
X_train_em, X_test_em, y_train_em, y_test_em = train_test_split(X_with_em_clusters, y_em, test_size=0.2, random_state=42)

# Initialize MLPClassifier for EM clusters
mlp_classifier_em = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)

# Train MLPClassifier with EM clusters
mlp_classifier_em.fit(X_train_em, y_train_em)

# Predict on the test set with EM clusters
y_pred_em = mlp_classifier_em.predict(X_test_em)

# Evaluate accuracy for EM clusters
accuracy_em = accuracy_score(y_test_em, y_pred_em)
print(f"Accuracy of MLPClassifier with EM clusters as features: {accuracy_em}")

# Step 2: Apply K-means clustering
n_clusters = 2  # Number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels_kmeans = kmeans.fit_predict(X)

# Append K-means cluster labels as new feature to the dataset
df_kmeans = df.copy()
df_kmeans['cluster_kmeans'] = cluster_labels_kmeans

# Split data into features with K-means clusters and target
X_with_kmeans_clusters = df_kmeans.drop('output', axis=1)
y_kmeans = df_kmeans['output']

# Split into training and testing sets with K-means clusters
X_train_kmeans, X_test_kmeans, y_train_kmeans, y_test_kmeans = train_test_split(X_with_kmeans_clusters, y_kmeans, test_size=0.2, random_state=42)

# Initialize MLPClassifier for K-means clusters
mlp_classifier_kmeans = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)

# Train MLPClassifier with K-means clusters
mlp_classifier_kmeans.fit(X_train_kmeans, y_train_kmeans)

# Predict on the test set with K-means clusters
y_pred_kmeans = mlp_classifier_kmeans.predict(X_test_kmeans)

# Evaluate accuracy for K-means clusters
accuracy_kmeans = accuracy_score(y_test_kmeans, y_pred_kmeans)
print(f"Accuracy of MLPClassifier with K-means clusters as features: {accuracy_kmeans}")