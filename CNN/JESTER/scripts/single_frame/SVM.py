# Copyright: (c) 2019, Bruno Calogero <brunocalogero@hotmail.com>
# GNU General Public License v3.0+ (see COPYING or https://www.gnu.org/licenses/gpl-3.0.txt)

import numpy
import os
import time

import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, metrics
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score


def load_JESTER(path):
    """
    Imports the NMNIST Dataset (right now hardcoded for n_Train_3 and n_Test_3 datasets (34x34x2 interleaved))
    """

    xs_train = []
    ys_train = []
    xs_test = []
    ys_test = []

    labels = ['Swiping_Down', 'Swiping_Left', 'Swiping_Right', 'Swiping_Up']

    for label in labels:

        for (root, dirs, dat_files) in os.walk('{0}/n_Train_interleaved/{1}'.format(path, label)):

            for file in dat_files:
                single_X = np.fromfile('{0}/n_Train_interleaved/{1}/{2}'.format(path, label, file), dtype=np.int32)
                xs_train.append(single_X)

                if label == 'Swiping_Down':
                    ys_train.append(0)
                elif label == 'Swiping_Up':
                    ys_train.append(1)
                elif label == 'Swiping_Left':
                    ys_train.append(2)
                elif label == 'Swiping_Right':
                    ys_train.append(3)

        for (root, dirs, dat_files) in os.walk('{0}/n_Test_interleaved/{1}'.format(path, label)):
            for file in dat_files:
                single_X_test = np.fromfile('{0}/n_Test_interleaved/{1}/{2}'.format(path, label, file), dtype=np.int32)
                xs_test.append(single_X_test)

                if label == 'Swiping_Down':
                    ys_test.append(0)
                elif label == 'Swiping_Up':
                    ys_test.append(1)
                elif label == 'Swiping_Left':
                    ys_test.append(2)
                elif label == 'Swiping_Right':
                    ys_test.append(3)

    Xtr = np.array(xs_train)
    Ytr = np.array(ys_train)
    Xte = np.array(xs_test)
    Yte = np.array(ys_test)

    # Sanity Check
    print('Type of Xtr:', type(Xtr))
    print('Type of Ytr:', type(Ytr[0]))
    print('Type of Xte:', type(Xte))
    print('Type of Yte:', type(Yte))

    return Xtr, Ytr, Xte, Yte


# NOTE: Pulling up the N-MNIST data
dataset_class_path = 'D:/LowPowerActionRecognition/CNN/JESTER/datasets'
X_train, Y_train, X_test, Y_test = load_JESTER(dataset_class_path)



# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', Y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', Y_test.shape)

# # shuffle the examples and their respective labels (training)
X_train, Y_train = shuffle(X_train, Y_train, random_state=1)
X_test, Y_test = shuffle(X_test, Y_test, random_state=1)


# Reshaping training and inference sets for CNN input
X_trainy = X_train.reshape(X_train.shape[0], 100, 176, 2)
X_testy = X_test.reshape(X_test.shape[0], 100, 176, 2)


# As a sanity check, we print out the size of the training and test data.
print('RESIZED DATA')
print('Training data shape: ', X_trainy.shape)
print('Training labels shape: ', Y_train.shape)
print('Test data shape: ', X_testy.shape)
print('Test labels shape: ', Y_test.shape)

# quick sanity check to make sure our data looks like what we want as a 2D array, thus meaning our 34x34x2 is correct
print(Y_train[105])
plt.imshow(X_trainy[105].reshape(100, 352))
plt.show()

X_train, X_val, Y_train, Y_val = train_test_split(X_trainy, Y_train, test_size=0.15, random_state=1)

# As a sanity check, we print out the size of the training and test data.
print('FINAL TRAIN/VAL/TEST split')
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', Y_train.shape)
print('validation data shape: ', X_val.shape)
print('validation labels shape: ', Y_val.shape)
print('Test data shape: ', X_testy.shape)
print('Test labels shape: ', Y_test.shape)


print('\n Printing a few labels from validation and training sets ')
print('validation:', Y_val[:10])
print('training:', Y_train[:10])


# Preprocessing: reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

# sanity check
print('X_train of shape:', X_train.shape)
print('X_val of shape:', X_val.shape)
print('X_test of shape:', X_test.shape)



# creating dummy SVM classifier for hyperparameterization
classifier = svm.SVC()

n_folds = 5
# choosing different parameter combinations to try
param_grid = {'C': [0.01, 0.1, 1, 10],
              'gamma': [0.00002, 0.0001, 0.001, 0.01],
              'kernel': ['rbf', 'linear'],
             }

# type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)


# run grid search
start_time = dt.datetime.now()
print('Start grid search at {}'.format(str(start_time)))

grid_search = GridSearchCV(classifier, param_grid, cv=n_folds, scoring=acc_scorer, n_jobs=4)
grid_obj = grid_search.fit(X_val, Y_val)
# get grid search results
print(grid_obj.cv_results_)

# set the best classifier found for rbf
clf = grid_obj.best_estimator_
print(clf)
end_time = dt.datetime.now()
print('Stop grid search {}'.format(str(end_time)))
elapsed_time= end_time - start_time
print('Elapsed grid search time {}'.format(str(elapsed_time)))


# fit the best alg to the training data
start_time = dt.datetime.now()
print('Start learning with best params at {}'.format(str(start_time)))

clf.fit(X_train, Y_train)

end_time = dt.datetime.now()
print('Stop learning {}'.format(str(end_time)))
elapsed_time= end_time - start_time
print('Elapsed learning time {}'.format(str(elapsed_time)))

# predict using test set
predictions = clf.predict(X_test)
print(accuracy_score(Y_test, predictions))

# Now predict the value of the test
expected = Y_test

print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(expected, predictions)))

cm = metrics.confusion_matrix(expected, predictions)
print("Confusion matrix:\n%s" % cm)

# plot_confusion_matrix(cm)

print("Accuracy={}".format(metrics.accuracy_score(expected, predictions)))
