import pandas as pd


def GroupEstimate(object):
    def __init__(self, estimate):
        valid_estimate = ['mean', 'median']
        if estimate.lower() not in valid_estimate:
            raise ValueError(f'Estimate must be one of {valid_estimate}')

        self.estimate = estimate.lower()
    
    def fit(self, X, y):
        return None

    def predict(self, X):
        return None