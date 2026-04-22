import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('your_dataset.csv')  # Update with your dataset path

# Prepare features and labels
X = df.drop('target', axis=1)  # Update 'target' with your label column
y = df['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_predictions)}")

# Support Vector Machine
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)
print(f"SVM Accuracy: {accuracy_score(y_test, svm_predictions)}")

# Gradient Boosting
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, y_train)
gb_predictions = gb_model.predict(X_test)
print(f"Gradient Boosting Accuracy: {accuracy_score(y_test, gb_predictions)}")
