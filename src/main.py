from data_util import Data
from data_util import plot_multiple_test_curves
from classifier import Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

DATA_DIR = "../data/"


def main():

    data_obj = Data(DATA_DIR)
  
    params_fact = {
        "n_estimators": 100,
        "max_depth": 10, 
        "criterion": "gini"
    }
    params_bias = {
        "n_estimators": 100,
        "max_depth": 10, 
        "criterion": "gini"
    }
    params_rf = {"fact": params_fact, "bias": params_bias}

    params_fact_sg = {
        "n_estimators": [1, 5, 10, 100],
        "max_depth": [1, 3, 5, 10], 
        "criterion": ["gini", "entropy"]
    }
    params_bias_sg = {
        "n_estimators": [1, 5, 10, 100],
        "max_depth": [1, 3, 5, 10], 
        "criterion": ["gini", "entropy"]
    }
    params_rf_sg = {"fact": params_fact_sg, "bias": params_bias_sg}


    params_fact = {
        "C": 0.1,
        "kernel": "rbf", 
        "degree": 3
    }
    params_bias = {
        "C": 0.1,
        "kernel": "rbf", 
        "degree": 3
    }
    params_svc = {"fact": params_fact, "bias": params_bias}

    params_fact_sg = {
        "C": [0.001, 0.1, 1.0, 10.0],
        "kernel": ["linear", "poly", "rbf"], 
        "degree": [1, 2, 3, 5]
    }
    params_bias_sg = {
        "C": [0.001, 0.1, 1.0, 10.0],
        "kernel": ["linear", "poly", "rbf"], 
        "degree": [1, 2, 3, 5]
    }
    params_svc_sg = {"fact": params_fact_sg, "bias": params_bias_sg}

    # Hyperparameters for Adaboost (Decision Tree - Stump) Classifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import AdaBoostClassifier

    ab_params_fact = {
        "base_estimator": DecisionTreeClassifier(max_depth=1), 
        "n_estimators": 100, 
        "learning_rate": 0.1
    }
    ab_params_bias = {
        "base_estimator": DecisionTreeClassifier(max_depth=1), 
        "n_estimators": 100, 
        "learning_rate": 0.1
    }
    params_ab = {"fact": ab_params_fact, "bias": ab_params_bias}

    ab_params_fact_sg = {
        "base_estimator": [DecisionTreeClassifier(max_depth=1)], 
        "n_estimators": [1, 3, 10, 100], 
        "learning_rate": [0.001, 0.1, 1.0, 10.0]
    }
    ab_params_bias_sg = {
        "base_estimator": [DecisionTreeClassifier(max_depth=1)], 
        "n_estimators": [1, 3, 10, 100], 
        "learning_rate": [0.001, 0.1, 1.0, 10.0]
    }
    params_ab_sg = {"fact": ab_params_fact_sg, "bias": ab_params_bias_sg}

    # Ridge Classifier
    from sklearn.linear_model import RidgeClassifier

    rc_params_fact = {
        "alpha": 1.0
    }
    rc_params_bias = {
        "alpha": 1.0
    }
    params_rc = {"fact": rc_params_fact, "bias": rc_params_bias}

    rc_params_fact_sg = {
        "alpha": [0.001, 0.01, 0.25, 0.5, 0.75, 1.25, 3.5, 10.0]
    }
    rc_params_bias_sg = {
        "alpha": [0.001, 0.01, 0.25, 0.5, 0.75, 1.25, 3.5, 10.0]
    }
    params_rc_sg = {"fact": rc_params_fact_sg, "bias": rc_params_bias_sg}

    # Adaboost (Decision Tree - Stumps) Classifier
    ab_clf = Classifier(AdaBoostClassifier, data_obj, params_ab, params_ab_sg, name="Adaboost - Decision Tree Stumps")
    ab_clf.sweep_params()
    ab_clf.create()
    ab_clf.fit()
    ab_lc_data = ab_clf.get_learning_curve_data()

    # Random Forest Classifier
    rf_clf = Classifier(RandomForestClassifier, data_obj, params_rf, params_rf_sg, name="Random Forest 1")
    rf_clf.sweep_params()
    rf_clf.create()
    rf_clf.fit()
    rf_lc_data = rf_clf.get_learning_curve_data())

    # Support Vector Classifier
    svc_clf = Classifier(SVC, data_obj, params_svc, params_svc_sg, name="Support Vector Classifier")
    svc_clf.sweep_params()
    svc_clf.create()
    svc_clf.fit()
    svc_lc_data = svc_clf.get_learning_curve_data()

    # Ridge Classifier
    rc_clf = Classifier(RidgeClassifier, data_obj, params_rc, params_rc_sg, name="Ridge Classifier")
    rc_clf.sweep_params()
    rc_clf.create()
    rc_clf.fit()
    rc_learning_curve_data = rc_clf.get_learning_curve_data()

    plot_multiple_test_curves([rf_lc_data, svc_lc_data, ab_lc_data, svc_lc_data])
    
    
if __name__=="__main__":
    main()