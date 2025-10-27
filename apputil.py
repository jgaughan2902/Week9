import pandas as pd
import numpy as np

class GroupEstimate(object):
    def __init__(self, estimate):
        '''
        Constructor for GroupEstimate class

        Parameters:
        estimate (str): 'mean' or 'median'

        Return value:
        No return value
        '''

        # Define what are valid estimate inputs
        valid_estimate = ['mean', 'median']

        # Raise an error if the estimate input isn't valid
        if estimate.lower() not in valid_estimate:
            raise ValueError(f'Estimate must be one of {valid_estimate}')

        # Assign the estimate to the 'self' object
        self.estimate = estimate.lower()
    
    def fit(self, X, y):
        if isinstance(y, list):
            y = np.array(y)

        self.features = list(X.columns)

        y_series = pd.Series(y, name = "target_value", index = X.index)
        data = pd.concat([X, y_series], axis = 1)

        agg_func = np.mean if self.estimate == 'mean' else np.median

        self.group_map = data.groupby(self.features)['target_value'].agg(agg_func).to_dict()
        
        return self

    def predict(self, X_):
        if isinstance(X_, (list, np.ndarray)):
            X_ = pd.DataFrame(X_, columns = self.features)

        group_keys = [tuple(row) for row in X_.values]

        predictions = [self.group_map.get(key, np.nan) for key in group_keys]

        predictions_array = np.array(predictions)

        missing_count = np.sum(np.isnan(predictions_array))

        if missing_count > 0:
            print(f'There are {missing_count} observation contained group(s) that were assigned to NaN')

        return(predictions_array)