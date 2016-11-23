from sklearn import tree
from sklearn import metrics
from load_dataset import get_datasets


#Generate tree metrics from from partitioned dataset and save a plotable file of the tree


x_train, y_train, x_validation, y_validation = get_datasets("data/sonar.txt")

#Initialize classifier
tree_clf = tree.DecisionTreeClassifier()
#Train classifier
tree_clf = tree_clf.fit(x_train, y_train)

#Prediction with training data
y_pred = tree_clf.predict(x_train)
#Compute error with training data
accuracy = metrics.accuracy_score(y_train, y_pred)
print("Train error:" + str(1-accuracy))

#Prediction with validation data
y_pred = tree_clf.predict(x_validation)
#Compute error with validation data
accuracy = metrics.accuracy_score(y_validation, y_pred)
print("Validation error:" + str(1-accuracy))
#Save plot file
tree.export_graphviz(tree_clf, out_file='tree.dot')
