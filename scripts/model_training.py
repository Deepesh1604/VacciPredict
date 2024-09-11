import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Load preprocessed data
X_train_scaled = pd.read_csv('../data/X_train_scaled.csv')
y_train = pd.read_csv('../data/y_train.csv')

# Logistic Regression Model Training
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train.values.ravel())  # Use ravel to convert y_train to 1D array

# Save the trained model
joblib.dump(model, '../models/saved_model.pkl')
