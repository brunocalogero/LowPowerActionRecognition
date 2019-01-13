"""
We recover the saved model for inference on the NMNIST interleaved data
"""

import numpy as np
import datetime as dt
import tensorflow as tf
import time
import os
import math
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

class Dataset(object):
    def __init__(self, X, y, batch_size, shuffle=False):
        """
        Construct a Dataset object to iterate over data X and labels y

        Inputs:
        - X: Numpy array of data, of any shape
        - y: Numpy array of labels, of any shape but with y.shape[0] == X.shape[0]
        - batch_size: Integer giving number of elements per minibatch
        - shuffle: (optional) Boolean, whether to shuffle the data on each epoch
        """
        assert X.shape[0] == y.shape[0], 'Got different numbers of data and labels'
        self.X, self.y = X, y
        self.batch_size, self.shuffle = batch_size, shuffle

    def __iter__(self):
        N, B = self.X.shape[0], self.batch_size
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
        return iter((self.X[i:i+B], self.y[i:i+B]) for i in range(0, N, B))

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

# NOTE: Setting clear separation between Training, Test and Validation Subsets
num_training = 59000
num_validation = 1000
num_test = 10000

mask = range(num_training, num_training + num_validation)
X_val = X_trainy[mask]
y_val = Y_train[mask]
mask = range(num_training)
X_trainy = X_trainy[mask]
Y_train = Y_train[mask]
mask = range(num_test)
X_testy = X_testy[mask]
Y_test = Y_test[mask]

# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_trainy.shape)
print('Test data shape: ', X_testy.shape)
print('Validation data shape: ', X_val.shape)

test_dset = Dataset(X_testy, Y_test, batch_size=64)


"""
Declaring Original Graph for simple forward-pass, added last layer softmax for prediction
"""
inputs = tf.placeholder(tf.float32, [None, 34, 34, 2])
labels = tf.placeholder(tf.int32, [None])

channel_1, channel_2, channel_3, num_classes =  64, 32, 16, 10
# consider using initializer for conv layers (variance scaling)

conv1 = tf.layers.conv2d(inputs, channel_1, (3, 3), padding='SAME', activation=tf.nn.leaky_relu)
bn1 = tf.layers.batch_normalization(conv1)
pool1 = tf.layers.max_pooling2d(bn1, 2, 2)

# maybe add a dropout/dropconnect at some point here

conv2 = tf.layers.conv2d(pool1, channel_2, (3, 3), padding='SAME', activation=tf.nn.leaky_relu)
bn2 = tf.layers.batch_normalization(conv2)
pool2 = tf.layers.max_pooling2d(bn2, 2, 2)

conv3 = tf.layers.conv2d(pool2, channel_3, (3, 3), padding='SAME', activation=tf.nn.leaky_relu)
bn3 = tf.layers.batch_normalization(conv3)
pool3 = tf.layers.max_pooling2d(bn3, 2, 2)

conv3_flattened = tf.layers.flatten(pool3)
fc = tf.layers.dense(conv3_flattened, num_classes)

# adding softmax classifier after fully connected dense layer for predictions
predictions = tf.nn.softmax(fc)
prediction_ids = tf.argmax(predictions, axis=1)


print('SESSION START')
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    saver =  tf.train.Saver()
    print('Restoring Model')
    saver.restore(sess, "saved_model/network_weights.ckpt")
    num_correct, num_samples = 0, 0
    for x_batch, y_batch in test_dset:

        feed_dict = {
            inputs: x_batch,
            labels: y_batch,
        }

        y_preds = sess.run(prediction_ids, feed_dict=feed_dict)
        num_samples += x_batch.shape[0]
        num_correct += (y_preds == y_batch).sum()

    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))
