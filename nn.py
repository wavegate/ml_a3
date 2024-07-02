import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

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

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Initialize MLPClassifier without dimensionality reduction
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)

# Measure training time
start_time = time.time()
mlp_classifier.fit(X_train, y_train)
training_time_mlp = time.time() - start_time

# Measure prediction time
start_time = time.time()
y_pred_mlp = mlp_classifier.predict(X_test)
prediction_time_mlp = time.time() - start_time

# Evaluate accuracy
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
print(f"Accuracy of MLPClassifier without dimensionality reduction: {accuracy_mlp}")
print(f"Training time: {training_time_mlp} seconds")
print(f"Prediction time: {prediction_time_mlp} seconds")

from sklearn.decomposition import PCA

# Initialize PCA
pca = PCA(n_components=5, random_state=42)

# Fit and transform on training set, transform test set
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Initialize MLPClassifier with PCA
mlp_classifier_pca = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)

# Measure training time
start_time = time.time()
mlp_classifier_pca.fit(X_train_pca, y_train)
training_time_pca = time.time() - start_time

# Measure prediction time
start_time = time.time()
y_pred_pca = mlp_classifier_pca.predict(X_test_pca)
prediction_time_pca = time.time() - start_time

# Evaluate accuracy
accuracy_pca = accuracy_score(y_test, y_pred_pca)
print(f"Accuracy of MLPClassifier with PCA: {accuracy_pca}")
print(f"Training time: {training_time_pca} seconds")
print(f"Prediction time: {prediction_time_pca} seconds")

from sklearn.decomposition import FastICA

# Initialize ICA
ica = FastICA(n_components=5, random_state=42)

# Fit and transform on training set, transform test set
X_train_ica = ica.fit_transform(X_train)
X_test_ica = ica.transform(X_test)

# Initialize MLPClassifier with ICA
mlp_classifier_ica = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)

# Measure training time
start_time = time.time()
mlp_classifier_ica.fit(X_train_ica, y_train)
training_time_ica = time.time() - start_time

# Measure prediction time
start_time = time.time()
y_pred_ica = mlp_classifier_ica.predict(X_test_ica)
prediction_time_ica = time.time() - start_time

# Evaluate accuracy
accuracy_ica = accuracy_score(y_test, y_pred_ica)
print(f"Accuracy of MLPClassifier with ICA: {accuracy_ica}")
print(f"Training time: {training_time_ica} seconds")
print(f"Prediction time: {prediction_time_ica} seconds")

from sklearn.random_projection import GaussianRandomProjection

# Initialize RP
rp = GaussianRandomProjection(n_components=5, random_state=42)

# Fit and transform on training set, transform test set
X_train_rp = rp.fit_transform(X_train)
X_test_rp = rp.transform(X_test)

# Initialize MLPClassifier with RP
mlp_classifier_rp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)

# Measure training time
start_time = time.time()
mlp_classifier_rp.fit(X_train_rp, y_train)
training_time_rp = time.time() - start_time

# Measure prediction time
start_time = time.time()
y_pred_rp = mlp_classifier_rp.predict(X_test_rp)
prediction_time_rp = time.time() - start_time

# Evaluate accuracy
accuracy_rp = accuracy_score(y_test, y_pred_rp)
print(f"Accuracy of MLPClassifier with RP: {accuracy_rp}")
print(f"Training time: {training_time_rp} seconds")
print(f"Prediction time: {prediction_time_rp} seconds")