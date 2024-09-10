from sklearn.metrics import classification_report, confusion_matrix
from data_preprocessing import load_and_preprocess_data
from joblib import load

def evaluate_model(model_path, data_path):
    # Load saved model
    model = load(model_path)
    
    # Load the data
    _, X_test, _, y_test = load_and_preprocess_data(data_path)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Generate evaluation metrics
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    evaluate_model('../models/saved_model.pkl', '../data/h1n1_vaccine_prediction.csv')
