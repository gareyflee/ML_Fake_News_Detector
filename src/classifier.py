import data_util as data
import numpy as np
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix  
import numpy as np
import sys

class Classifier():
    def __init__(self, clf_function, data, params, param_search_grid=None):
        self.data = data
        self.params = params
        self.best_params = None
        self.param_search_grid = param_search_grid
        self.clf_func = clf_function
    
    def create(self, classifier_function):
        if self.best_params is not None:
            params_fact = self.best_params["fact"]
            params_bias = self.best_params["bias"]
        else:
            params_fact = self.params["fact"]
            params_bias = self.params["bias"]
        self.clf_bias = self.clf_func(**params_fact)
        self.clf_fact = self.clf_func(**params_bias)

    def fit(self):
        print("Fitting Bias Model to ", self.data.X_train.shape[0], "test points with ", self.data.X_train.shape[1], "features.")  
        self.clf_bias.fit(self.data.X_train, self.data.y_train["bias"])
        self.clf_fact.fit(self.data.X_train, self.data.y_train["fact"])

    def predict(self):
        y_pred_fact = self.clf_fact.predict(self.data.X_test)
        y_pred_bias = self.clf_bias.predict(self.data.X_test)
        acc_fact = self.evaluate(y_pred_fact, self.data.y_test["fact"])
        acc_bias = self.evaluate(y_pred_bias, self.data.y_test["bias"])
        print("Accuracy\n\tBias: ", acc_bias, "\nFactuality: ", acc_fact)

    def evaluate(self, y_pred, y):
        return metrics.accuracy_score(y, y_pred)

    def sweep_params(self):
        assert self.param_search_grid is not None, "Error, no grid specified."
        clf_bias_search = GridSearchCV(self.clf_func(), self.param_search_grid["bias"])
        clf_fact_search = GridSearchCV(self.clf_func(), self.param_search_grid["fact"])
        X = np.vstack([self.data.X_train, self.data.X_val])
        y = {}
        for key in self.data.y.keys():
            y[key] = np.hstack([self.data.y_train[key], self.data.y_val[key]])
        clf_bias_search.fit(X, y["bias"])
        clf_fact_search.fit(X, y["fact"])
        print("\tBias Grid Search Results.")
        means = clf_bias_search.cv_results_['mean_test_score']
        stds = clf_bias_search.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf_bias_search.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print("\n\tFactuality Grid Search Results.")
        means = clf_fact_search.cv_results_['mean_test_score']
        stds = clf_fact_search.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf_bias_search.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))