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

def SVM(X_train, y_train, X_test):
  clf = svm.SVC(kernel = 'linear', C = 1, gamma = 1)
  clf.fit(X_train, y_train)
  predicted = clf.predict(X_test)
  weight = clf.coef_
  return predicted, weight

def feature_select_weight_svm(weight, X_train, X_test, y_train, y_test):
  accuracy_array = list()
  accuracy_array = np.array(accuracy_array)
  for n in range(2, 5):
    ind = np.argpartition(weight, -n)[-n:]
    X_new_train = X_train[:, ind]
    X_new_test = X_test[:, ind]
    clf = svm.SVC(kernel = 'linear', C = 1, gamma =1)
    clf = clf.fit(X_new_train, y_train)
    predicted = clf.predict(X_new_test)
    accuracy = metrics.accuracy_score(y_test, predicted)
    accuracy_array = np.append(accuracy_array, accuracy)
  n_array = np.arange(2, 5)
  plt.plot(n_array, accuracy_array)
  plt.xlim([0,6])
  plt.ylim([0.0, 1.2])
  plt.rcParams['font.size'] = 12
  plt.title('Accuracy vs. n(Feature Selected with Highest Weights)-SVM')
  plt.xlabel('Number of features(n)')
  plt.ylabel('Accuracy')
  plt.savefig('accuracy_weight_svm.png')


def feature_select_random_svm(weight, X_train, X_test, y_train, y_test):
  accuracy_array = list()
  accuracy_array = np.array(accuracy_array)
  for n in range(2, 5):
    #randomly generate the indices based on size m
    ind = np.random.randint(0,3,n)
    #create new train and test data
    X_new_train = X_train[:,ind]
    X_new_test = X_test[:,ind]
    #train on training data
    model = svm.SVC(kernel = 'linear', C = 1, gamma = 1)
    model = model.fit(X_new_train, y_train)
    #test on test data
    predicted = model.predict(X_new_test)
    accuracy = metrics.accuracy_score(y_test, predicted)
    accuracy_array = np.append(accuracy_array, accuracy)
  n_array = np.arange(2, 5)
  #do plot
  plt.plot(n_array, accuracy_array)  
  plt.xlim([0, 6])
  plt.ylim([0.0, 1.2])
  plt.rcParams['font.size'] = 12
  plt.title('Accuracy vs. n(Feature Selected Randomly)-SVM')
  plt.xlabel('Number of features(n)')
  plt.ylabel('Accuracy')
  plt.savefig('accuracy_random_svm.png')


