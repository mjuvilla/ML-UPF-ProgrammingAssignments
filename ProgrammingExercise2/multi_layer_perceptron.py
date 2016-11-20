from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from load_dataset import load_dataset
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

#Load dataset
x,y = load_dataset("data/mushrooms.txt")

#Declare error lists
error_trainList = []
error_validationList = []

#Plot error curves
def plot_err(list_train,list_validation):
    plt.plot(*zip(*list_train))
    plt.plot(*zip(*list_validation))
    plt.title('Error-alpha curves')
    plt.ylabel('Error')
    plt.xlabel('Alpha')
    #Logaritmic x_axe
    plt.xscale('log')
    red_label = mpatches.Patch(color='blue', label='train')
    green_label = mpatches.Patch(color='green', label='validation')
    plt.legend(handles=[red_label,green_label])
    plt.show()

#Scores
for c_alpha in np.logspace(-4,2,5):
    print "testing " + str(c_alpha)
    #Declare Multi-Layer-Perceptron classifier with L-2 Regularization (Weight-decay penalty)
    mlpclassifier = MLPClassifier(activation="logistic", alpha=c_alpha)
    #K-Fold partitioning
    kf = KFold(n_splits=10)
    #SDeclare score lists
    train_scores = []
    validation_scores = []
    #Perform prediction and compute mean of all val/train errors in all partitions
    for train, validation in kf.split(x):
        x_train, x_validation, y_train, y_validation = x[train], x[validation], y[train], y[validation]
        mlpclassifier.fit(x_train, y_train)
        train_scores.append(mlpclassifier.score(x_train, y_train))
        validation_scores.append(mlpclassifier.score(x_validation, y_validation))

    #print ("validation score= " + str(validation_scores))
    error_trainList.append((c_alpha,1-np.mean(train_scores)))
    error_validationList.append((c_alpha, 1 - np.mean(validation_scores)))



plot_err(error_trainList,error_validationList)





