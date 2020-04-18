import numpy as np
import matplotlib.pyplot as plt
# This file should contain all data processing functions

class Data:
    def __init__(self, X, y, train_len):
        assert(X.shape[0] == y.shape[0], "Error, number of labels does not match number of points.")
        self.X_all = np.array(X)
        self.y_all = np.array(y)
        if train_len <= 1.0:
            train_len = int(len(y) * train_len)
        else:
            assert(train_len < len(y), "Error, training set length must be less than total number of samples.")
        self.split_train_test()

    def split_train_test(self):
        pass

    def split_k(self, k):
        self.shuffle()
        self.x_k = np.array_split(self.X, k)
        self.y_k = np.array_split(self.y, k)

    def shuffle(self, X=None, y=None):
        # Shuffle the entire data set
        # If no arguments are given, all data in class is shuffled,
        # otherwise the given X and y are shuffled and returned
        if X is None:
            assert(y is None, "Error, Either supply no arguments or both data and labels.")
            rand_idxs = np.random.permutation(self.y.shape[0])
            self.X = self.X[rand_idxs]
            self.y = self.y[rand_idxs]
            return None
        else:
            assert(y is not None, "Error, Either supply no arguments or both data and labels.")
            rand_idxs = np.random.permutation(y.shape[0])
            return X[rand_idxs], y[rand_idxs]
            




def plot_accs(x_axis, train_acc, val_acc, plt_title="", x_title="Epochs", y_title="Accuracies", x_axis_log=False, save=False):
    assert(len(x_axis) == len(train_acc) == len(val_acc), "Error, x-axis and accuracies must be same length.")
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

def normalize(data):
    norm = np.linalg.norm(data, ord=2, axis=1)
    return (data.T / norm).T



def split_K(x, y):
