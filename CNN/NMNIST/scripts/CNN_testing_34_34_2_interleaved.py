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

def check_accuracy(sess, dset, x, scores, is_training=None):
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

def model_init_fn(inputs):
    """
    Defining our Tensorflow model (for the moment 4 different layers (including last FC layer))
    """
    channel_1, channel_2, channel_3, num_classes =  64, 32, 16, 10
    # consider using initializer for conv layers (variance scaling)

    conv1 = tf.layers.conv2d(inputs, channel_1, (3, 3), padding='SAME', activation=tf.nn.leaky_relu)
    bn1 = tf.layers.batch_normalization(conv1)
    pool1 = tf.layers.max_pooling2d(bn1, 2, 2)

    # maybe add a dropout at some point here

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
dataset_class_path = '/Users/brunocalogero/Desktop/LowPowerActionRecognition/CNN/NMNIST/datasets'
X_train, Y_train, X_test, Y_test = load_NMNIST(dataset_class_path)

# NOTE: Set up global variables For GPU use
USE_GPU = False

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
num_epochs = 20
# NOTE: constant to control the rate at which we save the model
save_every = 250


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

# Getting batches setup from Dataset Class
train_dset = Dataset(X_trainy, Y_train, batch_size=64, shuffle=True)
val_dset = Dataset(X_val, y_val, batch_size=64, shuffle=False)
test_dset = Dataset(X_testy, Y_test, batch_size=64)

# sanity check
print(train_dset)


# START TRAINING
tf.reset_default_graph()
with tf.device(device):
    x = tf.placeholder(tf.float32, [None, 34, 34, 2])
    y = tf.placeholder(tf.int32, [None])
    is_training = tf.placeholder(tf.bool, name='is_training')
    _ = tf.Variable(initial_value='fake_variable')

    scores = model_init_fn(x)
    loss   = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=scores)
    loss   = tf.reduce_mean(loss)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    t = 0
    for epoch in range(num_epochs):
        print('Starting epoch %d' % epoch)
        for x_np, y_np in train_dset:
            feed_dict = {x: x_np, y: y_np, is_training:1}
            loss_np, _ = sess.run([loss, train_op], feed_dict=feed_dict)
            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss_np))
                check_accuracy(sess, val_dset, x, scores, is_training=is_training)
            if t % save_every == 0:
                print('saving model..')
                save_path = tf.train.Saver().save(sess, "saved_model/network_weights.ckpt")
                print("Model saved in file: %s" % save_path)
            t += 1
    print('saving FINAL model..')
    save_path = tf.train.Saver().save(sess, "saved_model/network_weights.ckpt")
    print("FINAL Model saved in file: %s" % save_path)
