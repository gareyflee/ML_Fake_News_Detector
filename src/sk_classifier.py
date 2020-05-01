import numpy as np
import sys

from classifier import Classifier


class SK_Classifier(Classifier):
    def __init__(self, clf_function, data, params, param_search_grid=None):
        super().__init__(data, params)
        self.param_search_grid = param_search_grid
        self.clf_func = clf_function

    def create(self):
        super().create(self.clf_func)
      
    def fit(self):
        super().fit()
        
    def predict(self):
        super().predict()

    def evaluate(self, y_pred, y):
        acc = super().evaluate(y_pred, y)
        return acc