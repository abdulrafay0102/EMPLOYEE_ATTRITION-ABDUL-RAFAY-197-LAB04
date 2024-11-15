# EMPLOYEE_ATTRITION-ABDUL-RAFAY-197-LAB04
# ALL HE CODES ARE AVAILABLE IN THE FILE WHILE THE HOME TASK CODE IS:
#ABDUL RAFAY / 2022F-BSE-197 / LAB 04 / HOMETASK:
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif

# Load the data
file_path = 'extended_employee_data RAFAY.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("ABDUL RAFAY / 2022F-BSE-197 / LAB 04 / HOMETASK:\n")
print("Dataset Sample:")
print(data.head())

# Data Preprocessing
# Encode categorical variables
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])  # Male=1, Female=0
data['Position'] = le.fit_transform(data['Position'])
data['Attrition Status'] = le.fit_transform(data['Attrition Status'])  # Active=0, At Risk=1, Left=2

# Features and target variable
X = data.drop(columns=['Employee ID', 'Attrition Status'])
y = data['Attrition Status']

# Feature Engineering: Add Polynomial Features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

# Feature Selection: Use SelectKBest to choose the top features
k_best_selector = SelectKBest(score_func=f_classif, k=10)  # Selecting top 10 features
X_selected = k_best_selector.fit_transform(X_scaled, y)

# Split the dataset: Train with all except the last 10 rows, use the last 10 rows for testing
X_train, X_test = X_selected[:-10], X_selected[-10:]
y_train, y_test = y[:-10], y[-10:]

# Hyperparameter Tuning for Naive Bayes
# Use GridSearchCV to find the best 'var_smoothing' parameter
nb_model = GaussianNB()
param_grid = {'var_smoothing': np.logspace(0,-9, num=100)}
grid_search = GridSearchCV(nb_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best model found with GridSearchCV
best_nb_model = grid_search.best_estimator_

# Predictions using the best Naive Bayes model
y_pred_nb = best_nb_model.predict(X_test)

# Classification Report
print("Classification Report - Naive Bayes:")
print(classification_report(y_test, y_pred_nb))

# Predictions for the last 10 rows
print("Predictions for the last 10 rows - Naive Bayes:")
test_data_with_predictions_nb = data.tail(10).copy()
test_data_with_predictions_nb['Predicted Attrition Status (NB)'] = le.inverse_transform(y_pred_nb)
print(test_data_with_predictions_nb[['Employee ID', 'Attrition Status', 'Predicted Attrition Status (NB)']])

# Model Evaluation: Accuracy
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print("\n")
print(f"Naive Bayes Model Accuracy: {accuracy_nb * 100:.2f}%")

# Confusion Matrix and Classification Report for Naive Bayes
print("Confusion Matrix - Naive Bayes:")
print(confusion_matrix(y_test, y_pred_nb))
