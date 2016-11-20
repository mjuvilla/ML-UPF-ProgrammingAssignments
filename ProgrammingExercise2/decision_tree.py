from sklearn import tree
from sklearn import metrics
from load_dataset import get_datasets


#Generate tree metrics from first dataset partition
x_train, y_train, x_validation, y_validation = get_datasets("data/sonar.txt")

tree_clf = tree.DecisionTreeClassifier()
tree_clf = tree_clf.fit(x_train, y_train)

y_pred = tree_clf.predict(x_train)
accuracy = metrics.accuracy_score(y_train, y_pred)
print("Train error:" + str(1-accuracy))

y_pred = tree_clf.predict(x_validation)
accuracy = metrics.accuracy_score(y_validation, y_pred)
print("Validation error:" + str(1-accuracy))
tree.export_graphviz(tree_clf, out_file='tree2.dot')


