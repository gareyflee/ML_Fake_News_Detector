import numpy as np
import matplotlib.pyplot as plt
# This file should contain all data processing functions

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