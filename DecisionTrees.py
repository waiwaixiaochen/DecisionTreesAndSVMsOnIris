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


def decision_tree(X_train, y_train, X_test):
  clf = tree.DecisionTreeClassifier()
  clf = clf.fit(X_train, y_train)
  predicted = clf.predict(X_test)
#  score = clf.decision_function(X_test)
  return predicted

def accuracy(y_test, predicted):
  accuracy = metrics.accuracy_score(y_test, predicted)
  return accuracy

def feature_select_weight_tree(weight, X_train, X_test, y_train, y_test):
  accuracy_array = list()
  accuracy_array = np.array(accuracy_array)
  for n in range(2, 5):
    ind = np.argpartition(weight, -n)[-n:]
    X_new_train = X_train[:, ind]
    X_new_test = X_test[:, ind]
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_new_train, y_train)
    predicted = clf.predict(X_new_test)
    accuracy = metrics.accuracy_score(y_test, predicted)
    accuracy_array = np.append(accuracy_array, accuracy)
  n_array = np.arange(2, 5)
  plt.plot(n_array, accuracy_array)
  plt.xlim([0,6])
  plt.ylim([0.0, 1.2])
  plt.rcParams['font.size'] = 12
  plt.title('Accuracy vs. n(Feature Selected with Highest Weights)-DT')
  plt.xlabel('Number of features(n)')
  plt.ylabel('Accuracy')
  plt.savefig('accuracy_weight_tree.png')

def feature_select_random_tree(weight, X_train, X_test, y_train, y_test):
  accuracy_array = list()
  accuracy_array = np.array(accuracy_array)
  for n in range(2, 5):
    #randomly generate the indices based on size m
    ind = np.random.randint(0,3,n)
    #create new train and test data
    X_new_train = X_train[:,ind]
    X_new_test = X_test[:,ind]
    #train on training data
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_new_train, y_train)
    #test on test data
    predicted = clf.predict(X_new_test)
    accuracy = metrics.accuracy_score(y_test, predicted)
    accuracy_array = np.append(accuracy_array, accuracy)
  n_array = np.arange(2, 5)
  #do plot
  plt.plot(n_array, accuracy_array)  
  plt.xlim([0, 6])
  plt.ylim([0.0, 1.2])
  plt.rcParams['font.size'] = 12
  plt.title('Accuracy vs. n(Feature Selected Randomly)-DT')
  plt.xlabel('Number of features(n)')
  plt.ylabel('Accuracy')
  plt.savefig('accuracy_random_tree.png')



