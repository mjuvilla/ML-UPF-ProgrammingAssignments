from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from load_dataset import load_dataset
import matplotlib.pyplot as plt
import numpy as np

x, y = load_dataset("data/diabetes")

min_c = -1
max_c = 3
num_c = 50

min_gamma = -6
max_gamma = 1
num_gamma = 50

# create a list of c and gamma values to be evaluated
list_c, list_gamma = np.meshgrid(np.logspace(min_c, max_c, num_c),
                                 np.logspace(min_gamma, max_gamma, num_gamma))

error_parameters = list()

for c, gamma in zip(list_c.ravel(), list_gamma.ravel()):
    print "Evaluating C=" + str(c) + " and Gamma=" + str(gamma) + "..."
    svc = SVC(kernel="rbf", C=c, gamma=gamma)
    # get the 10-fold cross validation error
    scores = cross_val_score(svc, x, y, cv=10)
    #save c, gamma and error
    error_parameters.append((c, gamma, 1 - np.mean(scores)))

# get just the errors
error = zip(*error_parameters)[2]

# get the parameters that minimize the error
min_c, min_gamma, _ = error_parameters[np.argmin(error)]

error = np.array(error).reshape(list_c.shape)

print "Min C=" + str(min_c) + " Min Gamma= " + str(min_gamma)

# plot the error as a function of c and gamma
plt.title("Validation Error")
plt.contourf(list_c, list_gamma, error, cmap=plt.cm.coolwarm, alpha=0.8)
plt.xscale("log")
plt.yscale("log")
plt.xlabel('C')
plt.ylabel('Gamma')
plt.xlim(list_c.min(), list_c.max())
plt.ylim(list_gamma.min(), list_gamma.max())
plt.colorbar()
plt.show()

# train an SVM with the optimum parameters
svc = SVC(kernel="rbf", C=min_c, gamma=min_gamma)
svc = svc.fit(x, y)

# get the total error
y_pred = svc.predict(x)
accuracy = metrics.accuracy_score(y, y_pred)
print("Train error:" + str(1-accuracy))