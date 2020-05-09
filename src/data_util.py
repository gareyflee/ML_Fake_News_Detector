import numpy as np
import matplotlib.pyplot as plt
import os
import json
import codecs
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# This file should contain all data processing functions


class Data:
    '''
        Members:
            X - Design matrix with features of size nxd
            y - dictionary with keys "bias" and "fact" containing labels each with size nx1
            X_train, X_val, X_test - split data from X (default sizes in function argument)
            y_train, y_val, y_test - corresponding labels
            y_oh - one hot encoded matrix for "bias" and "fact" labels
            y_oh_train, y_oh_val, y_oh_test - corresponding one-hot encoded labels
    '''
    def __init__(self, DATA_DIR):
        print("\n----------- Loading Data -----------")
        self.labels = {}
        self.labels['fact'] = {'low': 0, 'mixed': 1, 'high': 2}
        self.labels['bias'] = {'extreme-right': 0, 'right': 1, 'right-center': 2, \
            'center': 3, 'left-center': 4, 'left': 5, 'extreme-left': 6}
        
        self.X, self.y = self.read_data(DATA_DIR)
        print("Data loaded with shapes:. \n\tFeatures Shape: ", self.X.shape, \
            "\n\tBias Labels Shape: ", self.y["bias"].shape, \
            "\n\tFactuality Labels Shape: ", self.y["fact"].shape)
        
        self.one_hot_encoding()
       
        self.split_train_test_val()
        self.split_k()
    
       
    def read_data(self, DATA_DIR):
        # Most of this code was taken from the original SVC paper
        data = pd.read_csv(DATA_DIR + "corpus.csv")
        self.sources = data.source_url_processed
        X = None

        for feature_file in os.listdir(DATA_DIR + 'features/'):
            if ".npy" in feature_file:
                feats = pd.DataFrame(np.load(DATA_DIR + 'features/' + feature_file, allow_pickle=True))
                feats = np.array(feats[feats.iloc[:, 0].isin(self.sources)])
                feats = np.delete(feats, 0, axis=1)
                feats = feats.astype(float)
                print("feets: ", feats.shape, feature_file)
                if X is None:
                    X = feats[:, :-2]
                else:
                    X = np.hstack([X, feats[:, :-2]])
                    print(X.shape)
            else:
                print(feature_file + " is not of type .npy, skipping.")
        y = {}
        y_bias = data["bias"]
        y["bias"] = np.array([self.labels["bias"][L.lower()] for L in y_bias])
        y_fact = data["fact"]
        y["fact"] = np.array([self.labels["fact"][L.lower()] for L in y_fact])
        
        return X, y
        
    def split_k(self, k=5):
        self.X_k = np.array_split(self.X, k, axis=0)
        self.y_k = {}
        self.y_oh_k = {}
        for key in self.y.keys():
            self.y_k[key] = np.array_split(self.y[key], k, axis=0)
            self.y_oh_k[key] = np.array_split(self.y_oh[key], k, axis=0)
        print("Data split into {} sets of approximate size: {}.".format(k, int(len(self.y["bias"])/k)))

    def one_hot_encoding(self):
        self.y_oh = {}
        label_encoder = LabelEncoder()
        onehot_encoder = OneHotEncoder(sparse=False)
        for key in self.y.keys():           
            integer_encoded = self.y[key].reshape(len(self.y[key]), 1)
            self.y_oh[key] = onehot_encoder.fit_transform(integer_encoded)
            # self.y_oh[key] = onehot_encoder.fit_transform(integer_encoded)
        print("One-hot-encoded labels generated.")


    def split_train_test_val(self, val_percent = 0.2, test_percent = 0.15):
        # Split all data into training and testing
        # If train len is not given, self.train_len param from constructor is used
        assert val_percent + test_percent < 1.0, "Error, percents are too high."
        train_len = int(self.X.shape[0] * (1.0 - val_percent - test_percent))
        test_len = int(self.X.shape[0] * test_percent)
        val_len = int(self.X.shape[0] * val_percent)

        self.shuffle()
        # Split the feature data
        self.X_train = self.X[:train_len, :]
        self.X_val = self.X[train_len:train_len + val_len, :]
        self.X_test = self.X[train_len + val_len:, :]

        # Split the label data
        self.y_train = {}
        self.y_test = {}
        self.y_val = {}
        self.y_oh_train = {}
        self.y_oh_test = {}
        self.y_oh_val = {}
        for key in self.y.keys():
            self.y_train[key] = self.y[key][:train_len]
            self.y_val[key] = self.y[key][train_len:train_len + val_len]
            self.y_test[key] = self.y[key][train_len + val_len:]
            self.y_oh_train[key] = self.y_oh[key][:train_len, :]
            self.y_oh_val[key] = self.y_oh[key][train_len:train_len + val_len, :]
            self.y_oh_test[key] = self.y_oh[key][train_len + val_len:, :]
        print("Train, Valiation, Test split complete.")

    def shuffle(self, X=None, y=None):
        # Shuffle the entire data set
        # If no arguments are given, all data in class is shuffled,
        # otherwise the given X and y are shuffled and returned
        print("Shuffling Data.")
        if X is None:
            assert y is None, "Error, Either supply no arguments or both data and labels."
            rand_idxs = np.random.permutation(self.y['fact'].shape[0])
            self.X = self.X[rand_idxs]
            for key in self.y.keys():
                self.y[key] = self.y[key][rand_idxs]
                self.y_oh[key] = self.y_oh[key][rand_idxs, :]
            return None
        else:
            assert y is not None, "Error, Either supply no arguments or both data and labels."
            rand_idxs = np.random.permutation(y['bias'].shape[0])
            return X[rand_idxs], y[rand_idxs]
            
def plot_multiple_test_curves(data_list, saveFig=True):
    colors = ["r", "g", 'b', 'c', 'm', 'y', 'k']
    type_labels = {"fact": "Factuality Classifier", "bias": "Bias Classifier"}
    for type in type_labels.keys():
        plt.figure()
        plt.grid()
        for i,clf_data in enumerate(data_list):
            print(clf_data)
            train_scores_mean = clf_data[type]["mean"]
            train_scores_std = clf_data[type]["std"]
            model_name = clf_data[type]["name"]
            train_sizes = clf_data[type]["sizes"]
            plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                train_scores_mean + train_scores_std, alpha=0.1, color=colors[i])
            plt.plot(train_sizes, train_scores_mean, 'o-',
                label=model_name, color=colors[i])
        plt.title(type_labels[type])
        plt.xlabel("Training Size")
        plt.ylabel("Validation Accuracy")
        plt.legend(loc="best")
        if saveFig:
            plt.savefig("./" + type_labels[type] + ".png")
        else:
            plt.show()
        plt.close()


def plot_iters(x_axis, train_acc, val_acc, train_std=None, val_std=None,\
        plt_title="", x_title="Epochs", y_title="Accuracies", x_axis_log=False, save=True, file_name=""):
    assert len(x_axis) == len(train_acc) == len(val_acc), "Error, x-axis and accuracies must be same length."
    fig = plt.figure()
    plt.plot(x_axis, train_acc, color="r", label="Training Accuracy")
    plt.plot(x_axis, val_acc, color="g", label="Validation Accuracy")
    if train_std is not None and val_std is not None:
        plt.fill_between(x_axis, train_acc-train_std, train_acc+train_std, alpha=0.1, color="r")
        plt.fill_between(x_axis, val_acc-val_std, val_acc+val_std, alpha=0.1, color="g")
    plt.legend()
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(plt_title)
    if x_axis_log:
        plt.xscale("log")
    if save:
        if file_name is None:
            file_name = plt_title.replace(" ", "_").replace(":", "-")
        plt.savefig("./" + file_name + ".png")
    else:
        plt.show()
    plt.close()
