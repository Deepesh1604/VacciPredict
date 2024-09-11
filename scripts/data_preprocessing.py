import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv('../data/h1n1_vaccine_prediction.csv')

# Identify categorical columns and perform One-Hot Encoding
categorical_columns = data.select_dtypes(include=['object']).columns
data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Features and target variable
X = data_encoded.drop(columns=['h1n1_vaccine'])
y = data_encoded['h1n1_vaccine']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the processed data
pd.DataFrame(X_train_scaled).to_csv('../data/X_train_scaled.csv', index=False)
pd.DataFrame(X_test_scaled).to_csv('../data/X_test_scaled.csv', index=False)
pd.DataFrame(y_train).to_csv('../data/y_train.csv', index=False)
pd.DataFrame(y_test).to_csv('../data/y_test.csv', index=False)
