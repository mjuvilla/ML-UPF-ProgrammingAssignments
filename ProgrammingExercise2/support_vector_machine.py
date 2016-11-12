from sklearn.svm import SVC
from sklearn import metrics
from load_dataset import get_datasets

x_train, y_train, x_validation, y_validation = get_datasets("data/letter.txt")

svc = SVC(kernel="rbf", C=10, gamma=0.01)