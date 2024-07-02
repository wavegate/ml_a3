import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load your dataset
df_stroke = pd.read_csv('stroke.csv')

# Preprocessing (as previously defined)
data = df_stroke.drop(columns=["id"])
data["bmi"].fillna(data["bmi"].mean(), inplace=True)

# Separate features and target variable
X_stroke = data.drop("stroke", axis=1)
y_stroke = data["stroke"]

# Identify categorical and numerical columns
categorical_cols_stroke = X_stroke.select_dtypes(include=["object"]).columns.tolist()
numerical_cols_stroke = X_stroke.select_dtypes(include=["number"]).columns.tolist()

# Preprocessing for numerical data
numerical_transformer_stroke = Pipeline(steps=[("scaler", StandardScaler())])

# Preprocessing for categorical data
categorical_transformer_stroke = Pipeline(
    steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
)

# Combine preprocessing steps
preprocessor_stroke = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer_stroke, numerical_cols_stroke),
        ("cat", categorical_transformer_stroke, categorical_cols_stroke),
    ]
)

# Preprocess the data
X_processed_stroke = preprocessor_stroke.fit_transform(X_stroke)
feature_names = (
    numerical_cols_stroke +
    list(preprocessor_stroke.named_transformers_['cat'].get_feature_names_out(categorical_cols_stroke))
)

# Create a DataFrame with processed data and feature names
X_processed_stroke = pd.DataFrame(X_processed_stroke, columns=feature_names)


# Save to CSV
X_processed_stroke['stroke'] = y_stroke
X_processed_stroke.to_csv('stroke_processed.csv', index=False)


print("Processed data saved to stroke_processed.csv")