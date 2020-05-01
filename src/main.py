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
        "n_estimators": [1, 2, 5],
        "max_depth": [1, 2, 5]
    }
    params_bias_sg = {
        "n_estimators": [1, 2, 5],
        "max_depth": [1, 2, 5]
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
        "degree": [1]
    }
    params_bias_sg = {
        "C": [0.001, 0.1, 1.0],
        "kernel": ["linear", "poly", "rbf"], 
        "degree": [1]
    }
    params_svc_sg = {"fact": params_fact_sg, "bias": params_bias_sg}

    # Hyperparameters for Adaboost (Decision Tree - Stump) Classifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import AdaBoostClassifier

    ab_params_fact = {
        "base_estimator": DecisionTreeClassifier(max_depth=1), 
        "n_estimators": 10, 
        "learning_rate": 1.
    }
    ab_params_bias = {
        "base_estimator": DecisionTreeClassifier(max_depth=1), 
        "n_estimators": 10, 
        "learning_rate": 1.
    }
    params_ab = {"fact": ab_params_fact, "bias": ab_params_bias}

    ab_params_fact_sg = {
        "base_estimator": [DecisionTreeClassifier(max_depth=1)], 
        "n_estimators": [1, 2, 5], 
        "learning_rate": [1.]
    }
    ab_params_bias_sg = {
        "base_estimator": [DecisionTreeClassifier(max_depth=1)], 
        "n_estimators": [1, 2, 5], 
        "learning_rate": [1.]
    }
    params_ab_sg = {"fact": ab_params_fact_sg, "bias": ab_params_bias_sg}

    # Adaboost (Decision Tree - Stumps) Classifier
    ab_clf = Classifier(AdaBoostClassifier, data_obj, params_ab, params_ab_sg, name="Adaboost - Decision Tree Stumps")
    ab_clf.sweep_params()
    ab_clf.create()
    ab_clf.fit()
    ab_clf.evaluate()


    rf_clf = Classifier(RandomForestClassifier, data_obj, params_rf, params_rf_sg, name="Random Forest")
    # rf_clf.create()
    # rf_clf.fit()
    # rf_clf.predict()
    rf_clf.sweep_params()
    rf_clf.create()
    rf_clf.fit()
    rf_clf.evaluate()

    return
    svc = Classifier(SVC, data_obj, params_svc, params_svc_sg, name="Support Vector Classifier")
    # svc.create()
    # svc.fit()
    # svc.evaluate()
    svc.sweep_params()
    svc.create()
    svc.fit()
    svc.evaluate()
    
if __name__=="__main__":
    main()