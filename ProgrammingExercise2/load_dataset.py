from sklearn.datasets import load_svmlight_file
import numpy as np

def load_dataset(filename):
    x, y = load_svmlight_file(filename)
    x_dense = x.todense()
    #y_matrix = np.matrix(y).transpose()
    return x_dense, y

def get_datasets(filename):
    X, Y = load_dataset(filename)

    total_size = len(Y)
    train_size = np.ceil(total_size * 0.75)

    train_indices = np.random.choice(total_size, train_size, replace=False)

    x_train = X[train_indices]
    y_train = Y[train_indices]

    x_validation = X[~train_indices]
    y_validation = Y[~train_indices]

    return x_train, y_train, x_validation, y_validation


