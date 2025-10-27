import pandas as pd
import numpy as np

class GroupEstimate(object):
    def __init__(self, estimate):
        valid_estimate = ['mean', 'median']
        if estimate.lower() not in valid_estimate:
            raise ValueError(f'Estimate must be one of {valid_estimate}')

        self.estimate = estimate.lower()
    
    def fit(self, X, y):
        if isinstance(y, list):
            y = np.array(y)

        self.features = list(X.columns)

        y_series = pd.Series(y, name = "target_value", index = X.index)
        data = pd.concat([X, y_series], axis = 1)

        agg_func = np.mean if self.estimate == 'mean' else np.median

        self.group_map = data.groupby(X_columns)['target_value'].agg(agg_func).to_dict()
        
        return self

    def predict(self, X_):
        if isinstance(X_, (list, np.ndarray)):
            X_ = pd.DataFrame(X_, columns = self.features)

        group_keys = [tuple(row) for row in X_.values]

        predictions = [self.group_map.get(key) for key in group_keys]

        return np.array(predictions)