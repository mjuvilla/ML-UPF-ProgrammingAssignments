from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from load_dataset import get_datasets

x_train, y_train, x_validation, y_validation = get_datasets("data/letter.txt")

mlpclassifier = MLPClassifier(activation="logistic")