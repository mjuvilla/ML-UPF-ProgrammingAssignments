from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt
import scipy.optimize
import numpy as np
import argparse
import time

def sigmoid(x):
    return (1 / (1 + np.exp(-x))) * 2 - 1

def load_dataset(filename):
    x_train, y_train = load_svmlight_file(filename)
    x_train_dense = x_train.todense()
    return x_train_dense, y_train

def get_subset(num_samples, x_train, y_train):
    if num_samples > y_train.shape[0]:
        num_samples = len(y_train)

    indices = np.random.choice(y_train.shape[0], num_samples, replace=False)
    return x_train[indices], y_train[indices]

def get_cost(w, x, y):
    return np.squeeze(np.asarray(np.log(1 / sigmoid(np.multiply(y.transpose() * x, w))).transpose() / len(y)))


def get_gradient(w, x, y):
    return np.squeeze(
        np.asarray(np.multiply(sigmoid(np.multiply(-y * x, w.transpose())), (-y * x)).transpose() / len(y)))

def train(x_train, y_train):
    w0 = np.random.rand(x_train.shape[1])
    w, cost, info = scipy.optimize.fmin_l_bfgs_b(get_cost, w0, get_gradient, args=(x_train, y_train))
    return np.array([w, ])

def inference(input, w):
    return sigmoid(np.squeeze(np.asarray(input * w.transpose())))

def compute_error(results, y_train):
    binarized_results = np.where(results > 0, 1., -1.)
    errors = 0
    for idx, label in enumerate(y_train):
        if label != binarized_results[idx]:
            errors += 1
    return errors / float(len(y_train))

def plot_w(weights):
    plt.bar(range(len(weights)), weights, align='center', alpha=0.5)
    plt.ylabel('Weight value')
    plt.xlabel('Feature number')
    plt.title('Weight vector representation')
    plt.show()

def plot_err(list):
    plt.plot(*zip(*list))
    plt.title('Error(#samples)')
    plt.ylabel('Aproximation Error')
    plt.xlabel('# of Samples')
    plt.show()

def plot_t(list):
    plt.plot(*zip(*list))
    plt.title('CPU_time(#samples)')
    plt.ylabel('Required CPU time')
    plt.xlabel('# of Samples')
    plt.show()

errorList = []
timeList = []

def main(filename, iterations):
    # inside args, we have the dataset_file attribute, which contains the filename of the selected dataset
    x_train, y_train = load_dataset(filename)

    # split the dataset in N iterations and take the ceiling (so we don't leave any sample out)
    samples_per_iteration = int(np.ceil(x_train.shape[0] / float(iterations)))

    for iteration in range(iterations):
        # increment the number of samples for this iteration
        # example: dataset of 100 samples, 10 iterations
        # first iteration: 10 samples; second iteration: 20 samples; etc.
        num_samples = samples_per_iteration + samples_per_iteration * iteration

        # sample the dataset
        x_sampled, y_sampled = get_subset(num_samples, x_train, y_train)
        # start time measure
        t0 = time.time()
        # compute the weights
        w = train(x_sampled, y_sampled)
        # compute required cpu-time for training
        t = time.time() - t0
        # append time elapsed into list
        timeList.append((len(y_sampled), t))
        # inference outputs the results given an input and some weights
        results = inference(x_sampled, w)
        # compute the error given the results and the ground truth
        error = compute_error(results, y_sampled)
        # append error data into list
        errorList.append((len(y_sampled), error))

        print("Num samples: " + str(len(y_sampled)) + ", Error: " + str(error) + ",Time: " + str(t))

        # Plot Weights for an iteration
        #plot_w(w)

    # Plot error curve
    plot_err(errorList)

    # Plot time curve
    plot_t(timeList)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # the script has two arguments, --dataset_file and --iterations
    # usage: python linear_regression.py --dataset_file datasets/housing_scale_little_regression --iterations 10
    parser.add_argument("--dataset_file", dest="dataset_file")
    parser.add_argument("--iterations", dest="iterations", type=int)
    # parse_args() puts all the function arguments into the "args" object
    args = parser.parse_args()
    main(args.dataset_file, args.iterations)



