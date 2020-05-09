import data_util as data
import numpy as np
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix  
import numpy as np
import sys
import json
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

class Classifier():
    # This class works for arbitrary classifiers as long as there is a fit and and evaluate method
    model_types = ["fact", "bias"]
    def __init__(self, clf_function, data, params, param_search_grid=None, name=""):
        self.data = data
        self.params = params
        self.best_params = None
        self.param_search_grid = param_search_grid
        self.clf_func = clf_function
        self.name = name
        
    def get_learning_curve_data(self):
        return self.learning_curve_data

    def plot_learning_curve(self, train_sizes = None):
        print("\n----------- Generating Learning Curve -----------")
        X = np.vstack([self.data.X_train, self.data.X_val])
        if train_sizes is None:
            train_sizes = np.linspace(start=0.1, stop=1, num=9)
        self.learning_curve_data = {}
        for type in self.model_types:
            print("Classifier Type: ", type)
            learning_curve_data = {}
            y = np.hstack([self.data.y_train[type], self.data.y_val[type]])
            cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
            train_sizes, train_scores, test_scores, fit_times, _ = \
                learning_curve(self.clf[type], X, y, cv=cv,  return_times=True, scoring="accuracy")
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)

            learning_curve_data["name"] = self.name
            if self.best_params is not None:
                learning_curve_data["params"] = self.best_params[type]
                learning_curve_data["is_grid_searched"] = True
                print("Grid search params found: ")
            else:
                learning_curve_data["params"] = self.params[type]
                learning_curve_data["is_grid_searched"] = False
            if "base_estimator" in learning_curve_data["params"].keys():
                learning_curve_data["params"]["base_estimator"] = "Decision Tree"
            learning_curve_data["mean"] = test_scores_mean
            learning_curve_data["std"] = test_scores_std
            learning_curve_data["sizes"] = train_sizes
            self.learning_curve_data[type] = learning_curve_data

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

    def fit(self):
        print("\n----------- Fitting Models -----------")
        print("Fitting Bias Model to ", self.data.X_train.shape[0], "test points with ", self.data.X_train.shape[1], "features.")  
        for type in self.model_types:
            print("Classifier Type: ", type)
            self.clf[type].fit(self.data.X_train, self.data.y_train[type])
        self.evaluate()
        self.plot_learning_curve()
    
    def predict(self):
        y_pred = {}
        for type in self.model_types:
            y_pred[type] = self.clf[type].predict(self.data.X_test)
        return y_pred

    def evaluate(self):
        print("\n----------- Evaluating Models -----------")
        y_pred = self.predict()
        acc = {}
        for type in self.model_types:
            print("Classifier Type: ", type)
            acc[type] = metrics.accuracy_score(self.data.y_test[type], y_pred[type])
        print_dict(acc)
        return acc

    def save_grid_search_results(self, clf_search, save=True):
        self.grid_search_plots = {}
        for type in self.model_types:
            self.grid_search_plots[type] = {}
            best_params = clf_search[type].best_params_
            print("Classifier Type: ", type, ", Params:\n", best_params)
            mean_test_scores = clf_search[type].cv_results_["mean_test_score"]
            mean_train_scores = clf_search[type].cv_results_["mean_train_score"]
            std_test_scores = clf_search[type].cv_results_['std_test_score']   
            std_train_score =  clf_search[type].cv_results_['std_train_score']
            all_params = clf_search[type].cv_results_['params']
            for plotting_param, plotting_val in best_params.items():
                const_params = best_params.copy()
                del const_params[plotting_param]
                if "base_estimator" in const_params.keys():
                    del const_params["base_estimator"]
                test_scores = np.array([score for i, score in enumerate(mean_test_scores) if (all(all_params[i][param]==value for param, value in const_params.items()))])
                train_scores = np.array([score for i, score in enumerate(mean_train_scores) if (all(all_params[i][param]==value for param, value in const_params.items()))])
                test_scores_std = np.array([score for i, score in enumerate(std_test_scores) if (all(all_params[i][param]==value for param, value in const_params.items()))])
                train_scores_std = np.array([score for i, score in enumerate(std_train_score) if (all(all_params[i][param]==value for param, value in const_params.items()))])
                
                param_sweep_vals= [val[plotting_param] for i, val in enumerate(all_params) if (all(all_params[i][param]==value for param, value in const_params.items()))]
                if len(param_sweep_vals) > 1:
                    filename = self.name + "_"
                    if type is "fact":
                        filename += "Factuality Classifier: "
                        title = "Factuality Classifier"
                    elif type is "bias":
                        filename += "Bias Classifier: "
                        title = "Bias Classifier"
                    for param, val in const_params.items():
                        filename += param + "= " + str(val) + ", "
                    filename = filename[:-2]
                    data.plot_iters(param_sweep_vals, train_scores, test_scores, train_scores_std, test_scores_std, save=save, plt_title=title, y_title="Accuracy", x_title=plotting_param, file_name=filename)

    def sweep_params(self, saveFig=True):
        assert self.param_search_grid is not None, "Error, no grid specified."
        print("\n----------- Performing Hyperparameter Grid Search -----------")
        X = np.vstack([self.data.X_train, self.data.X_val])
        y = {}
        clf_search = {}
        for type in self.model_types:
            print("Classifier Type: ", type)
            y[type] = np.hstack([self.data.y_train[type], self.data.y_val[type]])
            clf_search[type] = GridSearchCV(self.clf_func(), self.param_search_grid[type], \
                refit=True, return_train_score=True, verbose=1)
            clf_search[type].fit(X, y[type])
            print(type, " Grid Search Complete. Generating Plots.")
        self.save_grid_search_results(clf_search, save=saveFig)

def print_dict(dict, indent=2):
    print(json.dumps(dict, indent=indent))
