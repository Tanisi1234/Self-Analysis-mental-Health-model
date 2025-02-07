import pandas as pd
import numpy as np
import joblib
import lime.lime_tabular
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing

# Load dataset (Ensure train_df is correctly loaded before running this)
train_df = pd.read_csv("survey.csv")  # Uncomment this and set your file path

# Define features and target
feature_cols = ['Age', 'Gender', 'family_history', 'benefits', 'care_options', 'anonymity', 'leave', 'work_interfere']
X = train_df[feature_cols]
y = train_df['treatment']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Encoding categorical features
encoder_dict = {}
for col in ['Gender', 'family_history', 'benefits', 'care_options', 'anonymity', 'leave', 'work_interfere']:
    encoder = preprocessing.LabelEncoder()
    X[col] = encoder.fit_transform(X[col])  # Fit on full data
    encoder_dict[col] = encoder

# Apply encoding to train and test sets
X_train = X_train.copy()
X_test = X_test.copy()
for col in encoder_dict:
    X_train[col] = encoder_dict[col].transform(X_train[col])
    X_test[col] = encoder_dict[col].transform(X_test[col])

# Train Random Forest Model with Reduced Hyperparameter Search Space
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
rf_model = GridSearchCV(RandomForestClassifier(random_state=0), rf_params, cv=3, n_jobs=-1, verbose=1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.best_estimator_.predict(X_test)

# Train Neural Network Model with Reduced Hyperparameter Search Space
nn_params = {
    'hidden_layer_sizes': [(50, 50), (100, 50)],
    'activation': ['relu'],
    'solver': ['adam'],
    'alpha': [0.0001, 0.001],
    'max_iter': [500]
}
nn_model = GridSearchCV(MLPClassifier(random_state=0), nn_params, cv=3, n_jobs=-1, verbose=1)
nn_model.fit(X_train, y_train)
y_pred_nn = nn_model.best_estimator_.predict(X_test)

# Model Evaluation Function
def evaluate_model(y_test, y_pred, model_name):
    print(f"Evaluation for {model_name}:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, pos_label='Yes'))
    print("Recall:", recall_score(y_test, y_pred, pos_label='Yes'))
    print("F1 Score:", f1_score(y_test, y_pred, pos_label='Yes'))

    # Convert y_test and y_pred to numerical format for ROC-AUC calculation
    y_test_numeric = y_test.map({'Yes': 1, 'No': 0}).astype(int)  # Convert 'Yes' to 1, 'No' to 0
    y_pred_numeric = [1 if pred == 'Yes' else 0 for pred in y_pred] # Convert predictions to numeric

    print("ROC-AUC:", roc_auc_score(y_test_numeric, y_pred_numeric))
    print("\n")

# Evaluate Models
evaluate_model(y_test, y_pred_rf, "Random Forest")
evaluate_model(y_test, y_pred_nn, "Neural Network")

# Save trained models
joblib.dump(rf_model.best_estimator_, 'random_forest_model.pkl')
joblib.dump(nn_model.best_estimator_, 'neural_network_model.pkl')

# Save encoders for future use
joblib.dump(encoder_dict, 'label_encoders.pkl')

# Load models for testing
rf_model_loaded = joblib.load('random_forest_model.pkl')
nn_model_loaded = joblib.load('neural_network_model.pkl')

# Test Predictions with a Sample Input
sample_input = np.array([[30, 1, 1, 0, 1, 2, 0, 1]])  # Adjust based on actual dataset encoding

rf_sample_prediction = rf_model_loaded.predict(sample_input)
nn_sample_prediction = nn_model_loaded.predict(sample_input)

print("Sample Prediction - Random Forest:", rf_sample_prediction)
print("Sample Prediction - Neural Network:", nn_sample_prediction)

# LIME Explanation for Random Forest
lime_explainer_rf = lime.lime_tabular.LimeTabularExplainer(
    X_train.values, feature_names=feature_cols, class_names=['No', 'Yes'], discretize_continuous=True
)
exp_rf = lime_explainer_rf.explain_instance(X_test.iloc[0].values, rf_model_loaded.predict_proba, num_features=len(feature_cols))
exp_rf.show_in_notebook()

# LIME Explanation for Neural Network
lime_explainer_nn = lime.lime_tabular.LimeTabularExplainer(
    X_train.values, feature_names=feature_cols, class_names=['No', 'Yes'], discretize_continuous=True
)
exp_nn = lime_explainer_nn.explain_instance(X_test.iloc[0].values, nn_model_loaded.predict_proba, num_features=len(feature_cols))
exp_nn.show_in_notebook()
