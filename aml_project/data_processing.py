import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """ Load data from a CSV file """ 
        self.data = pd.read_csv(self.file_path)
        return self.data

    def clean_data(self):
        """ Clean the data by handling missing values and duplicates """ 
        self.data.dropna(inplace=True)  # Drop missing values
        self.data.drop_duplicates(inplace=True)  # Drop duplicates

    def normalize_data(self):
        """ Normalize the features in the data """ 
        scaler = StandardScaler()
        self.data[self.data.select_dtypes(include=['float64', 'int64']).columns] = scaler.fit_transform(
            self.data.select_dtypes(include=['float64', 'int64']).values)

    def split_data(self, target, test_size=0.2):
        """ Split the data into training and testing sets """ 
        X = self.data.drop(columns=[target])
        y = self.data[target]
        return train_test_split(X, y, test_size=test_size, random_state=42)

# Example usage:
# processor = DataProcessor('data.csv')
# processor.load_data()
# processor.clean_data()
# processor.normalize_data()
# X_train, X_test, y_train, y_test = processor.split_data(target='label')