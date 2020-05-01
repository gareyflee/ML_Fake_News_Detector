import data_util as data
import numpy as np
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix  
import numpy as np
import sys
import json
import matplotlib.pyplot as plt

class Classifier():
    model_types = ["fact", "bias"]
    def __init__(self, clf_function, data, params, param_search_grid=None, name=""):
        self.data = data
        self.params = params
        self.best_params = None
        self.param_search_grid = param_search_grid
        self.clf_func = clf_function
        self.name = name
        
    
    def create(self):
        print("\n----------- Building {} Classifiers -----------".format(self.name))
        params = {}
        if self.best_params is not None:
            print("Building {} models with parameters found in grid search.".format(self.name))
            params = self.best_params
            print("Accuracies: ")
            print_dict(self.best_score)
            self.clf = self.best_clf
        else:
            print("Building {} models with parameters given in constructor.".format(self.name))
            params = self.params
      
            self.clf = {}
            for type in self.model_types:
                param = params[type]
                print(param)
                print(self.params)
                self.clf[type] = self.clf_func(**param)
        # print("Parameters: ")
        # print_dict(params)

    def fit(self):
        print("\n----------- Fitting Models -----------")
        print("Fitting Bias Model to ", self.data.X_train.shape[0], "test points with ", self.data.X_train.shape[1], "features.")  
        for type in self.model_types:
            self.clf[type].fit(self.data.X_train, self.data.y_train[type])
    
    def predict(self):
        y_pred = {}
        for type in self.model_types:
            y_pred[type] = self.clf[type].predict(self.data.X_test)
        return y_pred
        

    def evaluate(self):
        y_pred = self.predict()
        acc = {}
        for type in self.model_types:
            acc[type] = metrics.accuracy_score(self.data.y_test[type], y_pred[type])
        print_dict(acc)
        return acc
    # def plot_param_sweep(self):
    #     assert self.best_params is not None, "Error, grid search has not been performed."
    #     print(self.best_clf.cv_results_)
        

    def sweep_params(self):
        assert self.param_search_grid is not None, "Error, no grid specified."
        print("\n----------- Performing Hyperparameter Grid Search -----------")
        # print_dict(self.param_search_grid, indent=2)
        X = np.vstack([self.data.X_train, self.data.X_val])
        y = {}
        clf_search = {}
        for type in self.model_types:
            y[type] = np.hstack([self.data.y_train[type], self.data.y_val[type]])
            clf_search[type] = GridSearchCV(self.clf_func(), self.param_search_grid[type], \
                refit=True, return_train_score=True, verbose=1)
            clf_search[type].fit(X, y[type])
            
        self.best_clf  = {}
        self.best_params = {}
        self.best_score = {}
        self.results = {}
        for type in self.model_types:
            self.best_clf[type] = clf_search[type].best_estimator_
            self.best_params[type] = clf_search[type].best_params_
            self.best_score[type] = clf_search[type].best_score_
        # print("Best Hyperparameters and Score:")
        # print_dict(self.best_params.update(self.best_score), indent=2)
def print_dict(dict, indent=2):
    print(json.dumps(dict, indent=indent))
  