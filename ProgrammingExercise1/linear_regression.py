from sklearn.datasets import load_svmlight_file
import numpy as np
from sklearn import linear_model


X_train, y_train = load_svmlight_file("datasets/housing_scale_little_regression.txt")
X_train_dense = X_train.todense()
y_train_matrix = np.matrix(y_train).transpose()
W = np.linalg.pinv(X_train_dense) * y_train_matrix

""""reg = linear_model.LinearRegression()
reg.fit(X_train,y_train)
error2=np.mean((reg.predict(X_train)-y_train) ** 2)"""""

results=X_train_dense*W
err1= np.power(results-y_train_matrix, 2)
error=(1/float(results.shape[0]))*np.sum(err1)
