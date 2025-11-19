import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib
import os

def load_data():
    print("Loading Olivetti faces dataset...")
    faces = fetch_olivetti_faces(shuffle=True, random_state=42)
    X = faces.data
    y = faces.target
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    return X, y

def split_data(X, y, test_size=0.3):
    print(f"Splitting data: {int((1-test_size)*100)}% train, {int(test_size*100)}% test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    print("Training DecisionTreeClassifier...")
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    print("Training completed!")
    return model

def save_model(model, filename='savedmodel.pth'):
    print(f"Saving model to {filename}...")
    joblib.dump(model, filename)
    print(f"Model saved successfully!")

def main():
    # Load data
    X, y = load_data()
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3)
    # Save test data for later evaluation
    joblib.dump((X_test, y_test), 'test_data.pkl')
    print("Test data saved for evaluation")
    # Train model
    model = train_model(X_train, y_train)
    # Save model
    save_model(model)
    print("\nTraining pipeline completed successfully!")

if __name__ == "__main__":
    main()