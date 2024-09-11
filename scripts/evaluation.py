import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load preprocessed data and trained model
X_test_scaled = pd.read_csv('../data/X_test_scaled.csv')
y_test = pd.read_csv('../data/y_test.csv')
model = joblib.load('../models/saved_model.pkl')

# Predict on test data
y_pred = model.predict(X_test_scaled)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
