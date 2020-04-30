import numpy as np
import matplotlib.pyplot as plt
import os
import json
import codecs
import argparse
import numpy as np
import pandas as pd
# This file should contain all data processing functions
temp_features = ["body", "title"]
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
        X = pd.DataFrame(X)
        y = {}
        y_bias = data["bias"]
        y["bias"] = np.array([self.labels["bias"][L.lower()] for L in y_bias])
        y_fact = data["fact"]
        y["fact"] = np.array([self.labels["fact"][L.lower()] for L in y_fact])
        return X, y

    def split_train_test(self, train_len=None):
        # Split all data into training and testing
        # If train len is not given, self.train_len param from constructor is used
        if train_len is None:
            train_len = self.train_len
        self.shuffle()
        self.x_train = self.X[:train_len, :]
        self.x_val = self.X[train_len:, :]
        self.y_train = self.y[:train_len]
        self.y_val = self.y[train_len:]

    def get_features_and_labels(corpus, features, task):
        # This function was taken from original model in classification.py
        data = pd.read_csv("data/corpus.csv")
        sources = data.source_url_processed
        X = np.empty(data.shape[0]).reshape(-1, 1)

    def split_k(self, k):
        # Split data into k sets and save them as their own members of the class
        self.shuffle()
        self.x_k = np.array_split(self.X, k)
        self.y_k = np.array_split(self.y, k)

    def shuffle(self, X=None, y=None):
        # Shuffle the entire data set
        # If no arguments are given, all data in class is shuffled,
        # otherwise the given X and y are shuffled and returned
        if X is None:
            assert y is None, "Error, Either supply no arguments or both data and labels."
            rand_idxs = np.random.permutation(self.y.shape[0])
            self.X = self.X[rand_idxs]
            self.y = self.y[rand_idxs]
            return None
        else:
            assert y is not None, "Error, Either supply no arguments or both data and labels."
            rand_idxs = np.random.permutation(y.shape[0])
            return X[rand_idxs], y[rand_idxs]
            
    def normalize(self, X=None, order=2):
        # Normalize data matrix
        # If no argument for matrix is gen, all data in class is shuffled, 
        # otherwise the shuffled array is returned
        if X is None:
            self.X = (self.X.T / np.linalg.norm(self.X, ord=order, axis=1)).T
            return None
        else:
            return (X.T / np.linalg.norm(X, ord=order, axis=1)).T

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

def parse_params():
    """
    Summary of the different tasks:
    -------------------------------
    fact:       {low, mixed, high}
    bias:       {extreme-right, right, center-right, center, center-left, left, extreme-left}
    bias3way:   {{extreme-right, right}, {center-right, center, center-left}, {left, extreme-left}}
    ===============================================================================================
    Summary of features from the different sources:
    -----------------------------------------------
    traffic:    alexa
    url:        handcrafted_url
    twitter:    has_twitter, created_at, verified, location, url_match, counts, description
    wikipedia:  has_wiki, wikicontent, wikisummary, wikitoc, wikicategories
    articles:   body, title
    =======================================================================================
    """
    parser = argparse.ArgumentParser(description='Source Reliability')
    parser.add_argument('--corpus',             type=str, default='MBFC_v2')
    parser.add_argument('--task',               type=str, default='bias')
    parser.add_argument('--features',           type=str, default='body+title') # list of features must be separated by "+" sign
    params = parser.parse_args()
    return params

def main():

    user_params = parse_params()
    corpus_filename = DATA_DIR + "corpus.csv"
    data_obj = Data(corpus_filename)

if __name__ == "__main__":
    main()