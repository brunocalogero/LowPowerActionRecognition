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


# Function to create model, required for KerasClassifier
def create_model(optimizer='adam', learn_rate=0.001, activation=tf.nn.leaky_relu):

    channel_1, channel_2, channel_3, num_classes =  64, 32, 8, 4
    # create model
    model = Sequential()

    model.add(Conv2D(channel_1, (3, 3), padding='SAME', activation=activation,  input_shape=(176, 100, 2), data_format="channels_last"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))

    model.add(Dropout(0.2))

    model.add(Conv2D(channel_2, (3, 3), padding='SAME', activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))

    # model.add(Conv2D(channel_3, (3, 3), padding='SAME', activation=activation))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    optimizer = Adam(lr=learn_rate)
    # Compile model (sparse cross-entropy can be used if one hot encoding not used)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[keras.metrics.categorical_accuracy])

    return model


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
X_trainy = X_train.reshape(X_train.shape[0], 176, 100, 2)
X_testy = X_test.reshape(X_test.shape[0], 176, 100, 2)


# As a sanity check, we print out the size of the training and test data.
print('RESIZED DATA')
print('Training data shape: ', X_trainy.shape)
print('Training labels shape: ', Y_train.shape)
print('Test data shape: ', X_testy.shape)
print('Test labels shape: ', Y_test.shape)

# quick sanity check to make sure our data looks like what we want as a 2D array, thus meaning our 34x34x2 is correct
print(Y_train[105])
plt.imshow(X_trainy[105].reshape(176, 200))
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


# # declaring number of folds for cross_validation
# n_folds = 8   # if using hyperparameterisation, please uncomment
epochs = 15
batch_size = 32

# retrieve model
model = create_model()
model.summary()

# fit the best alg to the training data
start_time = dt.datetime.now()
print('Start learning with best params at {}'.format(str(start_time)))

model.fit(X_train, to_categorical(Y_train, num_classes=4), validation_data=(X_val, to_categorical(Y_val, num_classes=4)), epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[TensorBoard(log_dir='tf_logs/1/train')])

end_time = dt.datetime.now()
print('Stop learning {}'.format(str(end_time)))
elapsed_time= end_time - start_time
print('Elapsed learning time {}'.format(str(elapsed_time)))


predictions_scores = model.predict(X_testy, batch_size=batch_size)

print(len(predictions_scores) == len(Y_test))
print(predictions_scores)
print(Y_test)


# Actual Inference
predictions = model.predict_classes(X_testy, batch_size=batch_size)

print(accuracy_score(Y_test, predictions))

# Now predict the value of the test
expected = Y_test

print("Classification report for classifier:\n%s\n", metrics.classification_report(expected, predictions))

cm = metrics.confusion_matrix(expected, predictions)
print("Confusion matrix:\n%s" % cm)

print("Accuracy={}".format(metrics.accuracy_score(expected, predictions)))
