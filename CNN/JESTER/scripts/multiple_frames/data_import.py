import os
import os.path

import numpy as np
import datetime as dt
from sklearn.utils import shuffle
from keras.utils import to_categorical

from extractor import Extractor
from tqdm import tqdm


class Dataset():
    '''
    Pull the neuromorphic JESTER dataset with flexibility and provide additional processing functionality
    '''

    def __init__(self, path, num_train=-1, num_test=-1, image_shape=(100, 176, 2), seq_length=12):

        self.image_shape = image_shape
        self.num_train = num_train
        self.num_test = num_test
        self.seq_length = seq_length
        self.path = path


    def load_JESTER(self, train_test, categorical=False):
        """
        This class method imports the neuromorphic JESTER Dataset
        """

        xs = []
        ys = []

        labels = ['Swiping_Down', 'Swiping_Left', 'Swiping_Right', 'Swiping_Up']

        data = 'train' if train_test == 'train' else 'test'

        for label in labels:
            for (root, dirs, dat_files) in os.walk('{0}/n_{1}/{2}'.format(self.path, data, label)):
                for file in dat_files:

                    if file != '.DS_Store':

                        single_X = np.load('{0}/n_{1}/{2}/{3}'.format(self.path, data, label, file))
                        single_X_resh = single_X.reshape(12, 100, 176, 2)

                        xs.append(single_X_resh)

                        if label == 'Swiping_Down':
                            ys.append(0)
                        elif label == 'Swiping_Up':
                            ys.append(1)
                        elif label == 'Swiping_Left':
                            ys.append(2)
                        elif label == 'Swiping_Right':
                            ys.append(3)

        X = np.array(xs)
        Y = np.array(ys)

        if categorical:
            Y = to_categorical(Y)

        # shuffle imported examples with randome_state seed (avoids np.random usage)
        X_data, Y_data = shuffle(X, Y, random_state=0)

        # Sanity Check
        print('Type of X:', type(X_data))
        print('Type of Y:', type(Y_data))

        return X_data, Y_data


    def load_JESTER_features(self, train_test):
        """
        This class method generates extracted features for each video.
        """

        # Set defaults.
        data = 'train' if train_test == 'train' else 'test'

        # get the model.
        model = Extractor()

        # setting up progress bar
        total_num = 16725 if train_test == 'train' else 2008
        # Loop through data.
        pbar = tqdm(total=total_num)

        xs = []
        ys = []

        labels = ['Swiping_Down', 'Swiping_Left', 'Swiping_Right', 'Swiping_Up']

        for label in labels:
            # print('{0}/n_{1}/{2}'.format(self.path, data, label))
            for (root, dirs, dat_files) in os.walk('{0}/n_{1}/{2}'.format(self.path, data, label)):
                for file in dat_files:
                    if file != '.DS_Store':

                        # Get the path to the sequence for this video.
                        path = os.path.join(self.path, 'sequences', '{}'.format(data), file[:-4] + '-' + str(self.seq_length) + \
                            '-features')  # numpy will auto-append .npy

                        # Check if we already have it.
                        if os.path.isfile(path + '.npy'):
                            pbar.update(1)
                            continue
                        else:
                            single_X = np.load('{0}/n_{1}/{2}/{3}'.format(self.path, data, label, file))
                            single_X_resh = single_X.reshape(12, 100, 176, 2)

                            xs.append(single_X_resh)

                            if label == 'Swiping_Down':
                                ys.append(0)
                            elif label == 'Swiping_Up':
                                ys.append(1)
                            elif label == 'Swiping_Left':
                                ys.append(2)
                            elif label == 'Swiping_Right':
                                ys.append(3)

                            # Now loop through and extract features to build the sequence.
                            sequence = []
                            for frame in single_X_resh:
                                features = model.extract(frame)
                                sequence.append(features)

                            # Save the sequence.
                            np.save(path, sequence)
                            pbar.update(1)

        pbar.close()
        Y = np.array(ys)
        label_path = os.path.join(self.path, 'sequences', '{}'.format(data), 'squence_labels')
        np.save(label_path, Y)


    def load_JESTER_sequences(self, train_test, categorical=True):
        """
        This class method loads up the InceptionV3, feature sequences
        """

        xs = []
        ys = []

        # labels = ['Swiping_Down', 'Swiping_Left', 'Swiping_Right', 'Swiping_Up']

        data = 'train' if train_test == 'train' else 'test'

        for (root, dirs, dat_files) in os.walk('{0}/sequences/{1}'.format(self.path, data)):
            for file in dat_files:
                if file != 'sequence_labels.npy':

                    single_X = np.load('{0}/sequences/{1}/{2}'.format(self.path, data, file)) # shape: (12, 2048)
                    xs.append(single_X)

                    if 'Swiping_Down' in file:
                        ys.append(0)
                    elif 'Swiping_Up' in file:
                        ys.append(1)
                    elif 'Swiping_Left' in file:
                        ys.append(2)
                    elif 'Swiping_Right' in file:
                        ys.append(3)

        X = np.array(xs)
        Y = np.array(ys)

        # sanity shape check
        print(X.shape)
        print(Y.shape)


        if categorical:
            Y = to_categorical(Y)

        # shuffle imported examples with randome_state seed (avoids np.random usage)
        X_data, Y_data = shuffle(X, Y, random_state=0)

        # Sanity Check
        print('Type of X:', type(X_data))
        print('Type of Y:', type(Y_data))

        return X_data, Y_data


def main():
    # NOTE: Pulling up the N-JESTER (reduced) dataset
    dataset_class_path = 'D:/LowPowerActionRecognition/CNN/JESTER/data'
    data = Dataset(path=dataset_class_path)

    # NOTE: uncomment below for testing load n_jester function
    # start_time = dt.datetime.now()
    # print('Start data import {}'.format(str(start_time)))
    #
    # X_test, Y_test = data.load_JESTER('test', categorical=True)
    # print('X_test of shape:', X_test.shape)
    # print('Y_test of shape:', Y_test.shape)
    #
    #
    # X_train, Y_train = data.load_JESTER('train', categorical=True)
    # print('X_train of shape:', X_train.shape)
    # print('Y_train of shape:', Y_train.shape)
    #
    # end_time = dt.datetime.now()
    # print('Stop load data time {}'.format(str(end_time)))
    #
    # elapsed_time= end_time - start_time
    # print('Elapsed load data time {}'.format(str(elapsed_time)))
    #
    # # sanity check
    # print('Sanity Check for shuffling:')
    # print(Y_test[4])
    # print(Y_test[5])
    # print(Y_test[6])

    # NOTE: uncomment below for feature generation
    # data.load_JESTER_features('test')

    # NOTE: uncomment below for feature sequence loading testing
    X_test, y_test = data.load_JESTER_sequences('train')

    print('shape for x_test:', X_test.shape)
    print('shape for y_test:', y_test.shape)


if __name__ == '__main__':
    main()