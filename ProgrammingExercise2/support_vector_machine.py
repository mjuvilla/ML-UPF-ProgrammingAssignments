from sklearn.svm import SVC
from sklearn import metrics
from load_dataset import get_datasets
from sklearn.model_selection import cross_val_score
from load_dataset import load_dataset
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

x,y = load_dataset("data/sonar.txt")

for c in np.linspace(0.001,10,10):
    for gamma in np.linspace(0.001,10,10):
     svc = SVC(kernel="rbf", C=c, gamma=gamma)
     scores = cross_val_score(svc, x, y, cv=10)
     #error_List.append()

