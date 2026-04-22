import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


class FeatureEngineeringPipeline:
    def __init__(self, scaling_method='standard', n_components=None):
        self.scaling_method = scaling_method
        self.n_components = n_components
        self.pipeline = self.create_pipeline()

    def create_pipeline(self):
        if self.scaling_method == 'standard':
            scaler = StandardScaler()
        elif self.scaling_method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError('Unsupported scaling method!')

        steps = [('scaler', scaler)]

        if self.n_components is not None:
            steps.append(('pca', PCA(n_components=self.n_components)))

        return Pipeline(steps)

    def fit(self, X):
        self.pipeline.fit(X)

    def transform(self, X):
        return self.pipeline.transform(X)

    def fit_transform(self, X):
        return self.pipeline.fit_transform(X)


# Usage Example:
# fe_pipeline = FeatureEngineeringPipeline(scaling_method='minmax', n_components=2)
# X_transformed = fe_pipeline.fit_transform(X)
