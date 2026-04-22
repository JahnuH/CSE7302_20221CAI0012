# Main Orchestration Script for AML Project

# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_data(file_path):
    """Load dataset from a CSV file"""
    return pd.read_csv(file_path)


def feature_engineering(data):
    """Perform feature engineering"""
    # Example: Encoding categorical variables
    data = pd.get_dummies(data)
    return data


def train_model(X_train, y_train):
    """Train the model using Random Forest"""
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the model performance"""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f'Model Accuracy: {accuracy * 100:.2f}%')


def main():
    # Load data
    data = load_data('dataset.csv')  # Adjust the path as necessary
    
    # Feature engineering
    processed_data = feature_engineering(data)
    
    # Split data into features and target
    X = processed_data.drop('target', axis=1)  # Assume 'target' is the label column
    y = processed_data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model training
    model = train_model(X_train, y_train)
    
    # Model evaluation
    evaluate_model(model, X_test, y_test)


if __name__ == '__main__':
    main()