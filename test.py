import joblib
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def load_model(filename='savedmodel.pth'):
    print(f"Loading model from {filename}...")
    model = joblib.load(filename)
    print("Model loaded successfully!")
    return model

def load_test_data(filename='test_data.pkl'):
    print(f"Loading test data from {filename}...")
    X_test, y_test = joblib.load(filename)
    print(f"Test data loaded: {X_test.shape[0]} samples")
    return X_test, y_test

def evaluate_model(model, X_test, y_test):
    print("\nEvaluating model on test set...")
    # Make predictions
    y_pred = model.predict(X_test)
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{'='*50}")
    print(f"TEST ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'='*50}")
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    return accuracy

def main():
    # Load model
    model = load_model()
    # Load test data
    X_test, y_test = load_test_data()
    # Evaluate model
    accuracy = evaluate_model(model, X_test, y_test)
    print("\nTesting pipeline completed successfully!")

if __name__ == "__main__":
    main()
