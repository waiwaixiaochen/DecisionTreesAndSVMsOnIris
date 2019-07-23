import DecisionTrees
import SVMs
import csv
import numpy as np
from sklearn import datasets
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def accuracy(y_test, predicted):
  accuracy = metrics.accuracy_score(y_test, predicted)
  return accuracy


iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)
#print(np.size(X_train,0))

tree_predicted = DecisionTrees.decision_tree(X_train, y_train, X_test)
tree_accuracy = accuracy(y_test, tree_predicted)
print('Accuracy by Decision Tree = {}'.format(tree_accuracy))
print(confusion_matrix(y_test,tree_predicted))

svm_predicted, weight = SVMs.SVM(X_train, y_train, X_test)
svm_accuracy = accuracy(y_test, svm_predicted)
print('Accuracy by SVM = {}'.format(svm_accuracy))
print(confusion_matrix(y_test, svm_predicted))

weight = np.absolute(weight[0])
print('weight = {}'.format(weight))

SVMs.feature_select_weight_svm(weight, X_train, X_test, y_train, y_test)
#DecisionTrees.feature_select_weight_tree(weight, X_train, X_test, y_train, y_test)
#SVMs.feature_select_random_svm(weight, X_train, X_test, y_train, y_test)
#DecisionTrees.feature_select_random_tree(weight, X_train, X_test, y_train, y_test)
