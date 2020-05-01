from data_util import Data
from classifier import Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

DATA_DIR = "../data/"


def main():

    data_obj = Data(DATA_DIR)
  
   
    
    params_fact = {
        "n_estimators": 10,
        "max_depth": 5
    }
    params_bias = {
        "n_estimators": 10,
        "max_depth": 5
    }
    params_rf = {"fact": params_fact, "bias": params_bias}

    params_fact_sg = {
        "n_estimators": [1, 2, 5, 10, 50],
        "max_depth": [1, 2, 5, 10, 20]
    }
    params_bias_sg = {
        "n_estimators": [1, 2, 5, 10, 50],
        "max_depth": [1, 2, 5, 10, 20]
    }
    params_rf_sg = {"fact": params_fact_sg, "bias": params_bias_sg}


    params_fact = {
        "C": 1.0,
        "kernel": "rbf", 
        "degree": 3
    }
    params_bias = {
        "C": 1.0,
        "kernel": "rbf", 
        "degree": 3
    }
    params_svc = {"fact": params_fact, "bias": params_bias}

    params_fact_sg = {
        "C": [0.001, 0.1, 1.0],
        "kernel": ["linear", "poly", "rbf"], 
        "degree": [1, 2]
    }
    params_bias_sg = {
        "C": [0.001, 0.1, 1.0],
        "kernel": ["linear", "poly", "rbf"], 
        "degree": [1, 2]
    }
    params_svc_sg = {"fact": params_fact_sg, "bias": params_bias_sg}


    rf_clf = Classifier(RandomForestClassifier, data_obj, params_rf, params_rf_sg, name="Random Forest")
    # rf_clf.create()
    # rf_clf.fit()
    # rf_clf.predict()
    rf_clf.sweep_params()
    rf_clf.create()
    rf_clf.fit()
    rf_clf.evaluate()
    # rf_clf.predict()
    svc = Classifier(SVC, data_obj, params_svc, params_svc_sg, name="Support Vector Classifier")
    # svc.create()
    # svc.fit()
    # svc.evaluate()
    svc.sweep_params()
    svc.create()
    svc.fit()
    svc.predict()
    
if __name__=="__main__":
    main()