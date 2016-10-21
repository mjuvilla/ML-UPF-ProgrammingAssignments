from sklearn.datasets import load_svmlight_file
import numpy as np

X_train, y_train = load_svmlight_file("datasets/sonar_scale_little_classification.txt")

W=np.random.rand(X_train.shape[0], 1)

