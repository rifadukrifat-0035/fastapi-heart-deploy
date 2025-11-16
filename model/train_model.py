import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

# Ensure 'model' directory exists
os.makedirs('model', exist_ok=True)

# Load the dataset
try:
    data = pd.read_csv('data/heart.csv')
except FileNotFoundError:
    print("Error: 'data/heart.csv' not found.")
    print("Please download the dataset from Kaggle and place it in the 'data/' directory.")
    exit()

# Define features (X) and target (y)
# 'target' 1 = heart disease, 0 = no heart disease
FEATURES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]
TARGET = 'target'

X = data[FEATURES]
y = data[TARGET]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save the model
model_path = 'model/heart_model.joblib'
joblib.dump(model, model_path)

print(f"Model saved to {model_path}")