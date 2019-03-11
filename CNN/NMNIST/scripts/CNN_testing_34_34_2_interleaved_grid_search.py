# Copyright: (c) 2019, Bruno Calogero <brunocalogero@hotmail.com>
# GNU General Public License v3.0+ (see COPYING or https://www.gnu.org/licenses/gpl-3.0.txt)

import numpy
import os
import time
import keras

import datetime as dt
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Activation, Flatten, Input
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.utils import to_categorical

from sklearn import svm, metrics
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score

# Setting up GPU / CPU, set log_device_placement to True to see what uses GPU and what uses CPU
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, device_count = {'GPU': 1 , 'CPU': 1})
sess = tf.Session(config=config)
keras.backend.set_session(sess)


# NOTE: Set up global variables For GPU use (not used in this code, just informative)
USE_GPU = True

if USE_GPU:
    device = '/device:GPU:0'
else:
    device = '/cpu:0'

print('Using device: ', device)


def load_NMNIST(path):
    """
    Imports the NMNIST Dataset (right now hardcoded for n_Train_3 and n_Test_3 datasets (34x34x2 interleaved))
    """
    xs_train = []
    ys_train = []
    xs_test = []
    ys_test = []

    for class_index in range(0, 10):
        for (root, dirs, dat_files) in os.walk('{0}/n_Train_3/{1}'.format(path, str(class_index))):
            for file in dat_files:
                single_X = np.fromfile('{0}/n_Train_3/{1}/{2}'.format(path, str(class_index), file), dtype=np.int32)
                xs_train.append(single_X)
                ys_train.append(class_index)

        for (root, dirs, dat_files) in os.walk('{0}/n_Test_3/{1}'.format(path, str(class_index))):
            for file in dat_files:
                xs_test.append(np.fromfile('{0}/n_Test_3/{1}/{2}'.format(path, str(class_index), file), dtype=np.int32))
                ys_test.append(class_index)

    Xtr = np.array(xs_train)
    Ytr = np.array(ys_train)
    Xte = np.array(xs_test)
    Yte = np.array(ys_test)

    return Xtr, Ytr, Xte, Yte

# Function to create model, required for KerasClassifier
def create_model(optimizer='adam', learn_rate=0.001, activation=tf.nn.leaky_relu):

    channel_1, channel_2, channel_3, num_classes =  64, 32, 16, 10
    # create model
    model = Sequential()

    model.add(Conv2D(channel_1, (3, 3), padding='SAME', activation=activation,  input_shape=(34, 34, 2), data_format="channels_last"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))

    model.add(Dropout(0.5))

    model.add(Conv2D(channel_2, (3, 3), padding='SAME', activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))

    model.add(Dropout(0.5))


    model.add(Conv2D(channel_3, (3, 3), padding='SAME', activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))

    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    optimizer = Adam(lr=learn_rate)
    # Compile model (sparse cross-entropy can be used if one hot encoding not used)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[keras.metrics.mae, keras.metrics.categorical_accuracy, 'accuracy'])

    return model


# def optimization_func_lr_hyper(n_folds):
#     '''
#     This function finds the best hyperparameters among those given by the user for the chosen optimization function
#     '''
#     print('hyperparameterization: optimization function and learning rate')
#     # create model
#     model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=64 ,verbose=0)
#
#     # define the grid search parameters
#     optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adam']
#     learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
#     param_grid = dict(optimizer=optimizer, learn_rate=learn_rate)
#     grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=n_folds)
#     grid_result = grid.fit(X_train, Y_train, validation_data=(X_val, Y_val))
#     # summarize results
#     print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#     means = grid_result.cv_results_['mean_test_score']
#     stds = grid_result.cv_results_['std_test_score']
#     params = grid_result.cv_results_['params']
#     for mean, stdev, param in zip(means, stds, params):
#         print("%f (%f) with: %r" % (mean, stdev, param))


# def activation_hyper(n_folds):
#     '''
#     This function finds the best hyperparameters among those given by the user for the chosen learning rate
#     '''
#
#     print('hyperparameterization: activation function')
#     # create model
#     model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=64, verbose=0)
#
#     # define the grid search parameters
#     activation = ['softmax', 'relu', 'tanh', 'sigmoid', 'linear']
#     param_grid = dict(activation=activation)
#     grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=n_folds)
#     grid_result = grid.fit(X_train, y_train, validation_data=(X_val, y_val))
#     # summarize results
#     print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#     means = grid_result.cv_results_['mean_test_score']
#     stds = grid_result.cv_results_['std_test_score']
#     params = grid_result.cv_results_['params']
#     for mean, stdev, param in zip(means, stds, params):
#         print("%f (%f) with: %r" % (mean, stdev, param))

# NOTE (TBD): hyperparameterization: number of neurons per layer to be added (if time allows)
# NOTE: Pulling up the N-MNIST data
dataset_class_path = 'D:/LowPowerActionRecognition/CNN/NMNIST/datasets'
X_train, Y_train, X_test, Y_test = load_NMNIST(dataset_class_path)


# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', Y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', Y_test.shape)

# # shuffle the examples and their respective labels (training)
X_train, Y_train = shuffle(X_train, Y_train, random_state=0)
X_test, Y_test = shuffle(X_test, Y_test, random_state=0)


# turn X training values into (60000, 34, 34, 2)
X_trainy = X_train.reshape(60000, 34, 34, 2)
# turn X test values into (10000, 34, 34, 2)
X_testy = X_test.reshape(10000, 34, 34, 2)


# As a sanity check, we print out the size of the training and test data.
print('RESIZED DATA')
print('Training data shape: ', X_trainy.shape)
print('Training labels shape: ', Y_train.shape)
print('Test data shape: ', X_testy.shape)
print('Test labels shape: ', Y_test.shape)

# quick sanity check to make sure our data looks like what we want as a 2D array, thus meaning our 34x34x2 is correct
print(Y_train[1])
plt.imshow(X_trainy[1].reshape(34,68))
plt.show()

X_train, X_val, Y_train, Y_val = train_test_split(X_trainy, Y_train, test_size=0.20, random_state=1)

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


# # declaring number of folds for cross_validation
# n_folds = 8   # if using hyperparameterisation, please uncomment
epochs = 15
batch_size = 64
n_folds = 8


# uncomment if needed
# optimization_func_lr_hyper(n_folds)
# activation_hyper(n_folds)

# retrieve model
model = create_model()
model.summary()

# fit the best alg to the training data
start_time = dt.datetime.now()
print('Start learning with best params at {}'.format(str(start_time)))

model.fit(X_train, to_categorical(Y_train, num_classes=10), validation_data=(X_val, to_categorical(Y_val, num_classes=10)), epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[TensorBoard(log_dir='tf_logs/3/train')])

end_time = dt.datetime.now()
print('Stop learning {}'.format(str(end_time)))
elapsed_time= end_time - start_time
print('Elapsed learning time {}'.format(str(elapsed_time)))


predictions_scores = model.predict(X_testy, batch_size=batch_size)


print('Sanity Check, is length of predictions same as ground truth labels: ', len(predictions_scores) == len(Y_test))
# print(predictions_scores)
# print(Y_test)


# Actual Inference
predictions = model.predict_classes(X_testy, batch_size=batch_size)

print(accuracy_score(Y_test, predictions))

# Now predict the value of the test
expected = Y_test

print("Classification report for classifier:\n%s\n", metrics.classification_report(expected, predictions))

cm = metrics.confusion_matrix(expected, predictions)
print("Confusion matrix:\n%s" % cm)

print("Accuracy={}".format(metrics.accuracy_score(expected, predictions)))
