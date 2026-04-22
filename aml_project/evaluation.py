import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Function to evaluate model performance

def evaluate_model(y_true, y_pred, y_probs):
    """
    Evaluate the model and return metrics.
    
    Parameters:
    y_true : array-like, true labels
    y_pred : array-like, predicted labels
    y_probs : array-like, predicted probabilities
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_probs)
    
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC': auc
    }
    return metrics

# Example usage
if __name__ == '__main__':
    # Simulated data for demonstration
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_probs = np.array([0.1, 0.9, 0.4, 0.2, 0.8])
    
    # Call the evaluation function
    metrics = evaluate_model(y_true, y_pred, y_probs)
    
    # Print the metrics
    for metric, value in metrics.items():
        print(f'{metric}: {value:.4f}')