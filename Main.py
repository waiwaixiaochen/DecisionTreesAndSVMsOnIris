import DecisionTree.py
import SVM.py

iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)
#print(np.size(X_train,0))

tree_predicted = decision_tree(X_train, y_train, X_test)
tree_accuracy = accuracy(y_test, tree_predicted)
print('Accuracy by Decision Tree = {}'.format(tree_accuracy))
print(confusion_matrix(y_test,tree_predicted))

svm_predicted, weight = SVM(X_train, y_train, X_test)
svm_accuracy = accuracy(y_test, svm_predicted)
print('Accuracy by SVM = {}'.format(svm_accuracy))
print(confusion_matrix(y_test, svm_predicted))

weight = np.absolute(weight[0])
print('weight = {}'.format(weight))

feature_select_weight_svm(weight, X_train, X_test, y_train, y_test)
#feature_select_weight_tree(weight, X_train, X_test, y_train, y_test)
#feature_select_random_svm(weight, X_train, X_test, y_train, y_test)
#feature_select_random_tree(weight, X_train, X_test, y_train, y_test)
