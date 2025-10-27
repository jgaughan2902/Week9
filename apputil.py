import pandas as pd


class GroupEstimate(object):
    def __init__(self, estimate):
        valid_estimate = ['mean', 'median']
        if estimate.lower() not in valid_estimate:
            raise ValueError(f'Estimate must be one of {valid_estimate}')

        self.estimate = estimate.lower()
    
    def fit(self, X, y):
        X_columns = list(X.columns)

        y_series = pd.Series(y, name = "target_value", index = X.index)
        data = pd.concat([X, y_series], axis = 1)

        agg_func = np.mean if self.estimate == 'mean' else np.median

        self.group_map = data.groupby(X_columns)['target_value'].agg(agg_func).to_dict
        return self

    def predict(self, X):
        return None