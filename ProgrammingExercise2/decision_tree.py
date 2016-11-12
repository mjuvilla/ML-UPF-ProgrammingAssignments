from sklearn import tree
from sklearn import metrics
from load_dataset import get_datasets

x_train, y_train, x_validation, y_validation = get_datasets("data/letter.txt")

tree = tree.DecisionTreeClassifier()
tree = tree.fit(x_train, y_train)

y_pred = tree.predict(x_train)
accuracy = metrics.accuracy_score(y_train, y_pred)
print("Train error:" + str(1-accuracy))

y_pred = tree.predict(x_validation)
accuracy = metrics.accuracy_score(y_validation, y_pred)
print("Validation error:" + str(1-accuracy))
