import time
import os
import math


import numpy as np
import datetime as dt
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

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

def check_accuracy(sess, dset, x, scores, train_writer, t, is_training=None):
    """
    Check accuracy on a classification model.

    Inputs:
    - sess: A TensorFlow Session that will be used to run the graph
    - dset: A Dataset object on which to check accuracy
    - x: A TensorFlow placeholder Tensor where input images should be fed
    - scores: A TensorFlow Tensor representing the scores output from the
      model; this is the Tensor we will ask TensorFlow to evaluate.

    Returns: Nothing, but prints the accuracy of the model
    """
    num_correct, num_samples = 0, 0
    for x_batch, y_batch in dset:
        feed_dict = {x: x_batch, is_training: 0}
        scores_np = sess.run(scores, feed_dict=feed_dict)
        y_pred = scores_np.argmax(axis=1)
        num_samples += x_batch.shape[0]
        num_correct += (y_pred == y_batch).sum()
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))
    val_acc = tf.summary.scalar("val_accuracy", acc)
    train_writer.add_summary(val_acc, t)

def model_init_fn(inputs):
    """
    Defining our Tensorflow model (for the moment 4 different layers (including last FC layer))
    """
    channel_1, channel_2, channel_3, num_classes =  64, 32, 16, 4
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

    return fc


# NOTE: Pulling up the N-MNIST data
dataset_class_path = 'D:/LowPowerActionRecognition/CNN/JESTER/datasets'
X_train, Y_train, X_test, Y_test = load_JESTER(dataset_class_path)


# NOTE: Set up global variables For GPU use
USE_GPU = True

if USE_GPU:
    device = '/device:GPU:0'
else:
    device = '/cpu:0'

print('Using device: ', device)

# NOTE: constant to control the learning rate of the optimizer
learning_rate = 0.001
# NOTE: Constant to control how often we print when training models
print_every = 2
# NOTE: constant to control the number of epochs we want to train for
num_epochs = 40
# NOTE: constant to control the rate at which we save the model
save_every = 100
# NOTE: constant to control the batch_size
batch_len = 32


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


# Getting batches setup from Dataset Class
train_dset = Dataset(X_train, Y_train, batch_size=batch_len, shuffle=True)
val_dset = Dataset(X_val, Y_val, batch_size=batch_len, shuffle=False)
test_dset = Dataset(X_testy, Y_test, batch_size=batch_len)

# sanity check
print(train_dset)

# # START TRAINING
# tf.reset_default_graph()
# with tf.device(device):
#     x = tf.placeholder(tf.float32, [None, 100, 176, 2])
#     y = tf.placeholder(tf.int32, [None])
#     is_training = tf.placeholder(tf.bool, name='is_training')
#     _ = tf.Variable(initial_value='fake_variable')
#
#     scores = model_init_fn(x)
#     loss   = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=scores)
#     loss   = tf.reduce_mean(loss)
#
#     optimizer = tf.train.AdamOptimizer(learning_rate)
#     update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#     with tf.control_dependencies(update_ops):
#         train_op = optimizer.minimize(loss)
#
#
# with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
#     sess.run(tf.global_variables_initializer())
#     t = 0
#     for epoch in range(num_epochs):
#         print('Starting epoch %d' % epoch)
#         for x_np, y_np in train_dset:
#             feed_dict = {x: x_np, y: y_np, is_training:1}
#             loss_np, _ = sess.run([loss, train_op], feed_dict=feed_dict)
#             if t % print_every == 0:
#                 print('Iteration %d, loss = %.4f' % (t, loss_np))
#                 check_accuracy(sess, val_dset, x, scores, is_training=is_training)
#             if t % save_every == 0:
#                 print('saving model..')
#                 save_path = tf.train.Saver().save(sess, "saved_models/1/network_weights.ckpt")
#                 print("Model saved in file: %s" % save_path)
#             t += 1
#     print('saving FINAL model..')
#     save_path = tf.train.Saver().save(sess, "saved_models/1/network_weights.ckpt")
#     print("FINAL Model saved in file: %s" % save_path)


# START TRAINING
tf.reset_default_graph()
with tf.device(device):
    x = tf.placeholder(tf.float32, [None, 100, 176, 2], name='example')
    y = tf.placeholder(tf.int32, [None], name='example_label')
    is_training = tf.placeholder(tf.bool, name='is_training')
    _ = tf.Variable(initial_value='fake_variable')

    scores = model_init_fn(x)
    loss   = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=scores)
    loss   = tf.reduce_mean(loss)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss)

    # add scalar summary for cost tensor
    cost_scalar = tf.summary.scalar("cost", loss)


with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
    sess.run(tf.global_variables_initializer())

    train_writer = tf.summary.FileWriter( './tf_logs/1/train', sess.graph)

    t = 0
    for epoch in range(num_epochs):
        print('Starting epoch %d' % epoch)
        for x_np, y_np in train_dset:


            feed_dict = {x: x_np, y: y_np, is_training:1}
            cost_summary, loss_np, _ = sess.run([cost_scalar, loss, train_op], feed_dict=feed_dict)
            if t % print_every == 0:
                print('Iteration %d, training loss = %.4f' % (t, loss_np))
                train_writer.add_summary(cost_summary, t)
                check_accuracy(sess, val_dset, x, scores, train_writer, t,  is_training=is_training)
            if t % save_every == 0:
                print('saving model..')
                save_path = tf.train.Saver().save(sess, "saved_models/1/network_weights.ckpt")
                print("Model saved in file: %s" % save_path)

            t += 1
    print('saving FINAL model..')
    save_path = tf.train.Saver().save(sess, "saved_models/1/network_weights.ckpt")
    print("FINAL Model saved in file: %s" % save_path)
