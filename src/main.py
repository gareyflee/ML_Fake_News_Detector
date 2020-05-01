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
        "n_estimators": [1, 2, 3],
        "max_depth": [1, 2, 5, 10, 20]
    }
    params_bias_sg = {
        "n_estimators": [1, 2, 3],
        "max_depth": [1, 2, 5, 10, 20]
    }
    params_rf_sg = {"fact": params_fact_sg, "bias": params_bias_sg}


    rf_clf = Classifier(RandomForestClassifier, data_obj, params_rf, params_rf_sg)
    rf_clf.sweep_params()
    # rf_clf.create()
    # rf_clf.fit()
    # rf_clf.predict()


if __name__=="__main__":
    main()