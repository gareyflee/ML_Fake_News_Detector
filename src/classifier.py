import data_util as data

class Classifier():
    def __init__(self, params: dictionary of hyperparameters):
        # Params should be a dictionary of hyperparameters
        self.params = params
    def __str__(self):
        # Use this for custom printing by just saying print(Class_Obj)
        return ""

    def train(self, x_train: design matrix, y_train):
        # Use this to train the model
        pass

    def cros_validation(self, k, param: string, x, y):
        assert (param in self.params.keys(), "Error, specified parameter not instantiated in constructor.")
        # This function should implement k-fold cross validation on the specified param
        pass

    def predict(self, x):
        # This function should accept either a single test point or design matrix
        # (Make sure it works for both)
        pass