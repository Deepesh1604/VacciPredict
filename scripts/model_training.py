from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from data_preprocessing import load_and_preprocess_data

def train_model(filepath):
    # Load and preprocess the data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(filepath)
    
    # Initialize the classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predictions on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy * 100:.2f}%')
    
    # Save the model for future use
    joblib.dump(model, '../models/saved_model.pkl')
    
    return model

if __name__ == "__main__":
    model = train_model('../data/h1n1_vaccine_prediction.csv')
