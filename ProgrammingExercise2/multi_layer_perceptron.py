from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from load_dataset import load_dataset
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

x,y = load_dataset("data/sonar.txt")
error_trainList = []
error_validationList = []

def plot_err(list_train,list_validation):
    plt.plot(*zip(*list_train))
    plt.plot(*zip(*list_validation))
    plt.title('cosa')
    plt.ylabel('Cross-validation error')
    plt.xlabel('alpha')
    red_label = mpatches.Patch(color='red', label='train')
    green_label = mpatches.Patch(color='green', label='validation')
    plt.legend(handles=[red_label,green_label])
    plt.show()

for c_alpha in np.logspace(-5,-2,10):
    print "testing " + str(c_alpha)
    mlpclassifier = MLPClassifier(activation="logistic", alpha=c_alpha)

    kf = KFold(n_splits=10)
    train_scores = []
    validation_scores = []
    for train, validation in kf.split(x):
        x_train, x_validation, y_train, y_validation = x[train], x[validation], y[train], y[validation]
        mlpclassifier.fit(x_train, y_train)
        train_scores.append(mlpclassifier.score(x_train, y_train))
        validation_scores = mlpclassifier.score(x_validation, y_validation)

    error_trainList.append((c_alpha,1-np.mean(train_scores)))
    error_validationList.append((c_alpha, 1 - np.mean(validation_scores)))



plot_err(error_trainList,error_validationList)

#cerca de alpha optima
#entrenar tot el dataset amb aquella alpha
#escala logaritmica plot



