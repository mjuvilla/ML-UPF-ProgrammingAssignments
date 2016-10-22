from sklearn.datasets import load_svmlight_file
import numpy as np
import argparse
import scipy.optimize

def load_dataset(filename):
    x_train, y_train = load_svmlight_file(filename)
    x_train_dense = x_train.todense()
    y_train_matrix = np.matrix(y_train).transpose()
    return x_train_dense, y_train_matrix

def get_subset(num_samples, x_train, y_train):
    if num_samples > len(y_train):
        num_samples = len(y_train)

    indices = np.random.choice(len(y_train), num_samples, replace=False)
    return x_train[indices], y_train[indices]

def get_cost(w, x, y):
    # TODO
    pass

def get_gradient(w, x, y):
    # TODO
    pass

def train(x_train, y_train):
    # TODO:
    w0 = np.random.rand(x_train.shape[0], 1)
    return scipy.optimize.fmin_bfgs(get_cost, w0, get_gradient, args=(x_train, y_train))

def inference(input, w):
    # TODO:
    pass

def compute_error(results, y_train):
    # TODO:
    pass

def main(filename, iterations):
    # inside args, we have the dataset_file attribute, which contains the filename of the selected dataset
    x_train, y_train = load_dataset(filename)

    # split the dataset in N iterations and take the ceiling (so we don't leave any sample out)
    samples_per_iteration = int(np.ceil(len(y_train) / float(iterations)))

    for iteration in range(iterations):
        # increment the number of samples for this iteration
        # example: dataset of 100 samples, 10 iterations
        # first iteration: 10 samples; second iteration: 20 samples; etc.
        num_samples = samples_per_iteration + samples_per_iteration * iteration

        # sample the dataset
        x_sampled, y_sampled = get_subset(num_samples, x_train, y_train)
        # compute the weights
        w = train(x_sampled, y_sampled)
        # inference outputs the results given an input and some weights
        results = inference(x_sampled, w)
        # compute the error given the results and the ground truth
        error = compute_error(results, y_sampled)

        print("Num samples: " + str(len(y_sampled)) + ", Error: " + str(error))

        # TODO: Plot error - num_samples

        # TODO: Plot time - num_samples

        # TODO: Plot weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # the script has two arguments, --dataset_file and --iterations
    # usage: python linear_regression.py --dataset_file datasets/housing_scale_little_regression --iterations 10
    parser.add_argument("--dataset_file", dest="dataset_file")
    parser.add_argument("--iterations", dest="iterations", type=int)
    # parse_args() puts all the function arguments into the "args" object
    args = parser.parse_args()
    main(args.dataset_file, args.iterations)



