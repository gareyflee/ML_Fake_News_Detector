import numpy as np
import matplotlib.pyplot as plt
import os
import json
import codecs
import argparse
import numpy as np
import pandas as pd
# This file should contain all data processing functions

DATA_DIR = "../News-Media-Reliability/data/"
class Data:
    def __init__(self, corpus_filename):
        self.labels = {}
        self.labels['fact'] = {'low': 0, 'mixed': 1, 'high': 2}
        self.labels['bias'] = {'extreme-right': 0, 'right': 1, 'right-center': 2, \
            'center': 3, 'left-center': 4, 'left': 5, 'extreme-left': 6}
        
        self.X, self.y = self.read_data(corpus_filename)
        print("Data loaded. \n\tFeatures Shape: ", self.X.shape, \
            "\n\tBias Labels Shape: ", self.y["bias"].shape, \
            "\n\tFactuality Labels Shape: ", self.y["fact"].shape)
        self.split_train_test_val()


    def read_data(self, corpus_filename):
        data = pd.read_csv(corpus_filename)
        self.sources = data.source_url_processed
        X = np.empty(data.shape[0]).reshape(-1, 1)
        for feature_file in os.listdir(DATA_DIR + 'features/'):
            if ".npy" in feature_file:
                feats = pd.DataFrame(np.load(DATA_DIR + 'features/' + feature_file, allow_pickle=True))
                feats = np.array(feats[feats.iloc[:, 0].isin(self.sources)])
                feats = np.delete(feats, 0, axis=1)
                feats = feats.astype(float)
                X = np.hstack([X, feats[:, :-2]])
            else:
                print(feature_file + " is not of type .npy, skipping.")
        y = {}
        y_bias = data["bias"]
        y["bias"] = np.array([self.labels["bias"][L.lower()] for L in y_bias])
        y_fact = data["fact"]
        y["fact"] = np.array([self.labels["fact"][L.lower()] for L in y_fact])
        return X, y

    def split_train_test_val(self, val_percent = 0.2, test_percent = 0.15):
        # Split all data into training and testing
        # If train len is not given, self.train_len param from constructor is used
        assert val_percent + test_percent < 1.0, "Error, percents are too high."
        train_len = int(self.X.shape[0] * (1.0 - val_percent - test_percent))
        test_len = int(self.X.shape[0] * test_percent)
        val_len = int(self.X.shape[0] * val_percent)

        self.shuffle()
        # Split the feature data
        self.x_train = self.X[:train_len, :]
        self.x_val = self.X[train_len:train_len + val_len, :]
        self.x_test = self.X[train_len + val_len:, :]

        # Split the label data
        self.y_train = {}
        self.y_test = {}
        self.y_val = {}
        for key in self.y.keys():
            self.y_train[key] = self.y[key][:train_len]
            self.y_val[key] = self.y[key][train_len:train_len + val_len]
            self.y_test[key] = self.y[key][train_len + val_len:]

    def shuffle(self, X=None, y=None):
        # Shuffle the entire data set
        # If no arguments are given, all data in class is shuffled,
        # otherwise the given X and y are shuffled and returned
        if X is None:
            assert y is None, "Error, Either supply no arguments or both data and labels."
            rand_idxs = np.random.permutation(self.y['fact'].shape[0])
            self.X = self.X[rand_idxs]
            for key in self.y.keys():
                self.y[key] = self.y[key][rand_idxs]
            return None
        else:
            assert y is not None, "Error, Either supply no arguments or both data and labels."
            rand_idxs = np.random.permutation(y['bias'].shape[0])
            return X[rand_idxs], y[rand_idxs]
            

def plot_accs(x_axis, train_acc, val_acc, plt_title="", x_title="Epochs", y_title="Accuracies", x_axis_log=False, save=False):
    assert len(x_axis) == len(train_acc) == len(val_acc), "Error, x-axis and accuracies must be same length."
    fig = plt.figure()
    plt.plot(x_axis, train_acc, label="Training Accuracy")
    plt.plot(x_axis, val_acc, label="Validation Accuracy")
    plt.legend()
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(plt_title)
    if x_axis_log:
        plt.xscale("log")
    if save:
        file_name = plt_title.replace(" ", "_")
        plt.savefig(plt_title + ".png")
    else:
        plt.show()
    plt.close()

def main():


    corpus_filename = DATA_DIR + "corpus.csv"
    data_obj = Data(corpus_filename)
    print(data_obj.y_train["bias"])

if __name__ == "__main__":
    main()