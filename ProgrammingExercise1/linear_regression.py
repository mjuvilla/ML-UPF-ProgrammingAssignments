from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import argparse
import time


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

def train(x_train, y_train):
    return np.linalg.pinv(x_train) * y_train

def inference(input, w):
    return input * w

def compute_error(results, y_train):
    return np.mean(np.power(results - y_train, 2))

def plot_w(weights):
    plt.bar(range(len(weights)), weights, align='center', alpha=0.5)
    plt.ylabel('Weight value')
    plt.xlabel('Feature number')
    plt.title('Weight vector representation')
    plt.show()

def plot_err(list,list_mean,list_median):
    plt.plot(*zip(*list))
    plt.plot(*zip(*list_mean))
    plt.plot(*zip(*list_median))
    plt.title('Error(#samples) in LinReg')
    plt.ylabel('Aproximation Error')
    plt.xlabel('# of Samples')
    red_label = mpatches.Patch(color='red', label='median')
    green_label = mpatches.Patch(color='green', label='mean')
    plt.legend(handles=[green_label,red_label])
    plt.show()

def plot_t(list,list_mean,list_median):
    plt.plot(*zip(*list))
    plt.plot(*zip(*list_mean))
    plt.plot(*zip(*list_median))
    plt.title('CPU_time(#samples) in LinReg')
    plt.ylabel('Required CPU time')
    plt.xlabel('# of Samples')
    red_label = mpatches.Patch(color='red', label='median')
    green_label = mpatches.Patch(color='green', label='mean')
    plt.legend(handles=[green_label, red_label])
    plt.show()

errorList = []
timeList = []

error_mean = []
t_mean = []

error_median = []
t_median = []

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
        # start time measure
        t0 = time.time()
        # compute the weights
        w = train(x_sampled, y_sampled)
        # compute required cpu-time for training
        t = time.time() - t0
        # append time elapsed into list
        timeList.append((len(y_sampled), t))
        current_mean = np.mean(zip(*timeList)[1])
        current_median = np.median(zip(*timeList)[1])
        t_mean.append((len(y_sampled), current_mean))
        t_median.append((len(y_sampled),current_median))
        # inference outputs the results given an input and some weights
        results = inference(x_sampled, w)
        # compute the error given the results and the ground truth
        error = compute_error(results, y_sampled)
        # append error data into list
        errorList.append((len(y_sampled), error))
        current_mean = np.mean(zip(*errorList)[1])
        current_median = np.median(zip(*errorList)[1])
        error_mean.append((len(y_sampled), current_mean))
        error_median.append((len(y_sampled),current_median))

        print("Num samples: " + str(len(y_sampled)) + ", Error: " + str(error) + ",Time: " + str(t))

        # Plot weights in two iterations
        if iteration % int((iterations-1) / 2.) == 0 and iteration != 0:
            plot_w(w)

        if len(y_sampled) == x_train.shape[0]:
            break

        # Plot error curve
    plot_err(errorList,error_mean,error_median)

    # Plot time curve
    plot_t(timeList,t_mean,t_median)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # the script has two arguments, --dataset_file and --iterations
    # usage: python linear_regression.py --dataset_file datasets/housing_scale_little_regression --iterations 100
    parser.add_argument("--dataset_file", dest="dataset_file")
    parser.add_argument("--iterations", dest="iterations", type=int)
    # parse_args() puts all the function arguments into the "args" object
    args = parser.parse_args()
    main(args.dataset_file, args.iterations)