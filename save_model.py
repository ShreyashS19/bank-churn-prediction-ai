import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import ExtraTreesClassifier
import joblib
import json
import shap
import pickle

# -----------------------------
# Load and preprocess data
# -----------------------------
data = pd.read_csv('input/credit-card-customers/BankChurners.csv')

# Rename columns
old_names = data.columns
new_names = [
    'Clientnum', 'Attrition', 'Age', 'Gender', 'Dependent_count', 'Education',
    'Marital_Status', 'Income', 'Card_Category', 'Months_on_book',
    'Total_Relationship_Count', 'Months_Inactive', 'Contacts_Count',
    'Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio', 'Naive_Bayes_1',
    'Naive_Bayes_2'
]
data.rename(columns=dict(zip(old_names, new_names)), inplace=True)

# Select relevant features (drop Clientnum, Naive_Bayes cols)
features = [
    'Age', 'Gender', 'Dependent_count', 'Education', 'Marital_Status', 'Income',
    'Card_Category', 'Months_on_book', 'Total_Relationship_Count',
    'Months_Inactive', 'Contacts_Count', 'Credit_Limit', 'Total_Revolving_Bal',
    'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio', 'Attrition'
]
data = data[features]

# Split X and y
X = data.drop(columns=['Attrition'])
y = data['Attrition'].map({'Existing Customer': 0, 'Attrited Customer': 1})

# Define categorical and numerical columns
categorical_cols = ['Gender', 'Education', 'Marital_Status', 'Income', 'Card_Category']
numerical_cols = [
    'Age', 'Dependent_count', 'Months_on_book', 'Total_Relationship_Count',
    'Months_Inactive', 'Contacts_Count', 'Credit_Limit', 'Total_Revolving_Bal',
    'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio'
]

# -----------------------------
# Train/Test split for metrics
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# -----------------------------
# Preprocessing: OneHotEncoder + StandardScaler
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# -----------------------------
# Build pipeline
# -----------------------------
pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('smote', SMOTE(random_state=0)),
    ('classifier', ExtraTreesClassifier(n_estimators=305, random_state=42))
])

# Fit pipeline on training data
pipeline.fit(X_train, y_train)

# -----------------------------
# Compute metrics on test set
# -----------------------------
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_proba)
f1 = f1_score(y_test, y_pred)

metrics = {
    "accuracy": round(acc, 4),
    "roc_auc": round(roc, 4),
    "f1_score": round(f1, 4),
    "model": "ExtraTrees",
    "estimators": 305
}

with open('backend/model_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f" Metrics — Accuracy: {acc:.4f}, AUC-ROC: {roc:.4f}, F1: {f1:.4f}")

# -----------------------------
# Retrain on full data for production model
# -----------------------------
pipeline.fit(X, y)

# -----------------------------
# Save trained model
# -----------------------------
joblib.dump(pipeline, 'backend/model.pkl')

# -----------------------------
# Compute SHAP values
# -----------------------------
print(" Computing SHAP values (this may take a moment)...")

# Transform data through preprocessing (without SMOTE)
X_transformed = pipeline.named_steps['preprocessing'].transform(X)

# Get feature names after transformation
num_feature_names = numerical_cols
cat_feature_names = list(
    pipeline.named_steps['preprocessing']
    .named_transformers_['cat']
    .get_feature_names_out(categorical_cols)
)
all_transformed_feature_names = num_feature_names + cat_feature_names

# Create TreeExplainer on the classifier
explainer = shap.TreeExplainer(pipeline.named_steps['classifier'])

# Use a sample for SHAP computation (full dataset can be slow)
sample_size = min(500, len(X))
np.random.seed(42)
sample_indices = np.random.choice(len(X), sample_size, replace=False)
X_sample = X.iloc[sample_indices]
X_sample_transformed = pipeline.named_steps['preprocessing'].transform(X_sample)

shap_values_raw = explainer.shap_values(X_sample_transformed)

# For binary classification, handle different SHAP output formats
# shap_values_raw can be: list of [class0, class1], 3D array (samples, features, classes), or 2D
if isinstance(shap_values_raw, list):
    shap_vals = shap_values_raw[1]  # class 1 (churn)
elif isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 3:
    shap_vals = shap_values_raw[:, :, 1]  # class 1 (churn)
else:
    shap_vals = shap_values_raw

print(f"   SHAP values shape: {shap_vals.shape}")

# Map transformed feature SHAP values back to original feature names
# by summing SHAP values for one-hot encoded features
original_feature_names = numerical_cols + categorical_cols
n_num = len(numerical_cols)
n_cat_features_expanded = len(cat_feature_names)

# Build mapping: original feature -> column indices in transformed matrix
feature_index_map = {}
idx = 0
for col in numerical_cols:
    feature_index_map[col] = [idx]
    idx += 1
for col in categorical_cols:
    cat_cols_for_feature = [
        i for i, name in enumerate(all_transformed_feature_names)
        if name.startswith(col + '_')
    ]
    feature_index_map[col] = cat_cols_for_feature

# Aggregate SHAP values per original feature
shap_aggregated = np.zeros((shap_vals.shape[0], len(original_feature_names)))
for i, feat in enumerate(original_feature_names):
    cols = feature_index_map[feat]
    shap_aggregated[:, i] = shap_vals[:, cols].sum(axis=1)

# Save SHAP data
shap_data = {
    'shap_values': shap_aggregated,
    'feature_names': original_feature_names,
    'X_sample': X_sample.values,
    'X_sample_columns': list(X_sample.columns),
    'sample_indices': sample_indices.tolist()
}

with open('backend/shap_values.pkl', 'wb') as f:
    pickle.dump(shap_data, f)

with open('backend/feature_names.json', 'w') as f:
    json.dump(original_feature_names, f)

print(f" SHAP values computed for {sample_size} samples")

# -----------------------------
# Save example input/output
# -----------------------------
example_input = X.head(5)
example_input.to_csv('backend/example_input.csv', index=False)

example_output = example_input.copy()
example_output['Prediction'] = [
    'Not Churn' if pred == 0 else 'Churn'
    for pred in pipeline.predict(example_input)
]
example_output.to_csv('backend/example_output.csv', index=False)

print(" Model training complete. Model, metrics, and SHAP data saved in backend/")
