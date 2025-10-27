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

        # Define what are valid estimate inputs.
        valid_estimate = ['mean', 'median']

        # Raise an error if the estimate input isn't valid.
        if estimate.lower() not in valid_estimate:
            raise ValueError(f'Estimate must be one of {valid_estimate}')

        # Assign the estimate to the 'self' object.
        self.estimate = estimate.lower()
    
    def fit(self, X, y):
        '''
        Method that takes in data, merges it and modifies it.

        Parameters:
        X (pandas.DataFrame): Contains categorical data
        y (list or np.array): A list or array of data

        Return value:
        self: Technically not necessary to answer the question
        but included to make things look complete.
        '''
        # If y input is a list, convert it to an array.
        if isinstance(y, list):
            y = np.array(y)

        # Columns in X are assigned to the features of the object.
        self.features = list(X.columns)

        # Merging X and y together into one.
        y_series = pd.Series(y, name = "target_value", index = X.index)
        data = pd.concat([X, y_series], axis = 1)

        # Define the aggregate function depending on the estimate
        # assigned previously.
        agg_func = np.mean if self.estimate == 'mean' else np.median

        # Group by function using the aggregate function
        self.group_map = data.groupby(self.features)['target_value'].agg(agg_func).to_dict()
        
        return self

    def predict(self, X_):
        '''
        Determines which group the array of observations in X_
        fall into.

        Parameters:
        X_ (np.array or pandas.DataFrame): An array or dataframe
        of values to be used to produce the return value.

        Return value:
        predictions_array (np.array): An np.array of corresponding
        estimates for y.
        '''
        
        # If X_ is a list or array, make it a pandas dataframe.
        if isinstance(X_, (list, np.ndarray)):
            X_ = pd.DataFrame(X_, columns = self.features)

        # Create the group keys.
        group_keys = [tuple(row) for row in X_.values]

        # Create predictions.
        predictions = [self.group_map.get(key, np.nan) for key in group_keys]

        # Convert the predictions into an array.
        predictions_array = np.array(predictions)

        # Find the amount of missing category or categories.
        missing_count = np.sum(np.isnan(predictions_array))

        # If the count is more than zero, print a message.
        if missing_count > 0:
            print(f'There are {missing_count} observation contained group(s) that were assigned to NaN')

        return(predictions_array)