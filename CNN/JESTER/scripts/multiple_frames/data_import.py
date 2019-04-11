# Copyright: (c) 2019, Bruno Calogero <brunocalogero@hotmail.com>
# GNU General Public License v3.0+ (see COPYING or https://www.gnu.org/licenses/gpl-3.0.txt)

'''
To Do: clean up and improve code reuse
'''

import random
import threading
import os
import os.path

import numpy as np
import datetime as dt
from sklearn.utils import shuffle
from keras.utils import to_categorical

# from extractor import Extractor
# from tqdm import tqdm


class threadsafe_iterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.iterator)

def threadsafe_generator(func):
    """Decorator"""
    def gen(*a, **kw):
        return threadsafe_iterator(func(*a, **kw))
    return gen


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
        if (train_test == 'train') or (train_test == 'test'):
            data = 'train' if train_test == 'train' else 'test'
            train_testy = train_test
        else:
            print('Caution: User has selected a train or test set that is not named n_train or n_test')
            print('User has selected: n_{}'.format(train_test))
            data = train_test

            if 'train' in train_test:
                train_testy = 'train'
            elif 'test' in train_test:
                train_testy = 'test'

        # get the model.
        model = Extractor()

        # setting up progress bar
        if '10' in train_test:
            total_num = 35108 if train_testy == 'train' else 4817
        else:
            total_num = 16725 if train_testy == 'train' else 2008
        # Loop through data.
        pbar = tqdm(total=total_num)

        xs = []
        ys = []

        if '10' in train_test:
            labels = ['Swiping_Down', 'Swiping_Left', 'Swiping_Right', 'Swiping_Up',
                      'Sliding_Two_Fingers_Down', 'Sliding_Two_Fingers_Left', 'Sliding_Two_Fingers_Right', 'Sliding_Two_Fingers_Up',
                      'Zooming_In_With_Two_Fingers', 'Zooming_Out_With_Two_Fingers']
        else:
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

                            if  '10' in train_test:
                                if label == 'Swiping_Down':
                                    ys.append(0)
                                elif label == 'Swiping_Up':
                                    ys.append(1)
                                elif label == 'Swiping_Left':
                                    ys.append(2)
                                elif label == 'Swiping_Right':
                                    ys.append(3)
                                elif label == 'Sliding_Two_Fingers_Down':
                                    ys.append(4)
                                elif label == 'Sliding_Two_Fingers_Left':
                                    ys.append(5)
                                elif label == 'Sliding_Two_Fingers_Right':
                                    ys.append(6)
                                elif label == 'Sliding_Two_Fingers_Up':
                                    ys.append(7)
                                elif label == 'Zooming_In_With_Two_Fingers':
                                    ys.append(8)
                                elif label == 'Zooming_Out_With_Two_Fingers':
                                    ys.append(9)
                            else:
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
        label_path = os.path.join(self.path, 'sequences', '{}'.format(data), 'sequence_labels')
        np.save(label_path, Y)


    def load_JESTER_sequences(self, train_test, categorical=True):
        """
        This class method loads up the InceptionV3, feature sequences
        """

        xs = []
        ys = []

        if (train_test == 'train') or (train_test == 'test'):
            data = 'train' if train_test == 'train' else 'test'
        else:
            print('Caution: User has selected a train or test set that is not named n_train or n_test')
            print('User has selected: n_{}'.format(train_test))
            data = train_test

        for (root, dirs, dat_files) in os.walk('{0}/sequences/{1}'.format(self.path, data)):
            for file in dat_files:
                if file != 'sequence_labels.npy':

                    single_X = np.load('{0}/sequences/{1}/{2}'.format(self.path, data, file)) # shape: (12, 2048)
                    xs.append(single_X)

                    if  '10' in train_test:
                        if 'Swiping_Down' in file:
                            ys.append(0)
                        elif 'Swiping_Up' in file:
                            ys.append(1)
                        elif 'Swiping_Left' in file:
                            ys.append(2)
                        elif 'Swiping_Right' in file:
                            ys.append(3)
                        elif 'Sliding_Two_Fingers_Down' in file:
                            ys.append(4)
                        elif 'Sliding_Two_Fingers_Left' in file:
                            ys.append(5)
                        elif 'Sliding_Two_Fingers_Right' in file:
                            ys.append(6)
                        elif 'Sliding_Two_Fingers_Up' in file:
                            ys.append(7)
                        elif 'Zooming_In_With_Two_Fingers' in file:
                            ys.append(8)
                        elif 'Zooming_Out_With_Two_Fingers' in file:
                            ys.append(9)
                    else:
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

    @threadsafe_generator
    def load_generator(self, train_test, batch_size=32, num_classes=10, categorical=True, regeneration=True):
        """
        This class method exports batches of data in the form of generator by yielding.
        Used for fit_generator.
        """

        examples = {}
        examples_copy = {}

        if '10' in train_test:
            labels = ['Swiping_Down', 'Swiping_Left', 'Swiping_Right', 'Swiping_Up',
                      'Sliding_Two_Fingers_Down', 'Sliding_Two_Fingers_Left', 'Sliding_Two_Fingers_Right', 'Sliding_Two_Fingers_Up',
                      'Zooming_In_With_Two_Fingers', 'Zooming_Out_With_Two_Fingers']
        else:
            labels = ['Swiping_Down', 'Swiping_Left', 'Swiping_Right', 'Swiping_Up']

        if (train_test == 'train') or (train_test == 'test'):
            data = 'train' if train_test == 'train' else 'test'
        else:
            print('Caution: User has selected a train or test set that is not named n_train or n_test')
            print('User has selected: n_{}'.format(train_test))
            data = train_test

        for label in labels:
            # NOTE: make sure that your data files do not contain '.DS_Store'
            for (root, dirs, dat_files) in os.walk('{0}/n_{1}/{2}'.format(self.path, data, label)):
                # populate dictionary with all examples with labels as keys
                examples[label] = dat_files

        examples_copy = examples.copy()

        while True:

            xs = []
            ys = []


            for label in labels:

                temp = []
                # random.sample() proved to be problematic but it was because of lack of dict reset so might try again
                temp = [random.choice(examples[label]) for _ in range(int(batch_size/num_classes))]

                # retrieve actual data
                for file in temp:
                    if file != '.DS_Store':
                        single_X = np.load('{0}/n_{1}/{2}/{3}'.format(self.path, data, label, file))
                        single_X_resh = single_X.reshape(12, 100, 176, 2)
                        xs.append(single_X_resh)

                        if  '10' in train_test:
                            if 'Swiping_Down' in file:
                                ys.append(0)
                            elif 'Swiping_Up' in file:
                                ys.append(1)
                            elif 'Swiping_Left' in file:
                                ys.append(2)
                            elif 'Swiping_Right' in file:
                                ys.append(3)
                            elif 'Sliding_Two_Fingers_Down' in file:
                                ys.append(4)
                            elif 'Sliding_Two_Fingers_Left' in file:
                                ys.append(5)
                            elif 'Sliding_Two_Fingers_Right' in file:
                                ys.append(6)
                            elif 'Sliding_Two_Fingers_Up' in file:
                                ys.append(7)
                            elif 'Zooming_In_With_Two_Fingers' in file:
                                ys.append(8)
                            elif 'Zooming_Out_With_Two_Fingers' in file:
                                ys.append(9)
                        else:
                            if 'Swiping_Down' in file:
                                ys.append(0)
                            elif 'Swiping_Up' in file:
                                ys.append(1)
                            elif 'Swiping_Left' in file:
                                ys.append(2)
                            elif 'Swiping_Right' in file:
                                ys.append(3)

                # pop the temp elements from dict with given label as key
                examples[label] = list(set(examples[label]) - set(temp))

                # if examples is getting empty (toward the end of the epoch), regenerate it
                if regeneration:
                    # fix zooming out with two fingers problem
                    # if label == 'Zooming_Out_With_Two_Fingers':
                    #     if len(examples[label]) <= 16:
                    #         print('Zooming out causing issues')
                    #         examples.update({label: examples_copy[label]})
                    # else:
                    if len(examples[label]) <= 16:
                        print('Regenerating other labels')
                        examples = examples_copy.copy()
                else:
                    if len(examples[label]) <= 16:
                        break
            # turn into numpy array
            X = np.array(xs)
            Y = np.array(ys)

            # one-hot label conversion
            if categorical:
                Y = to_categorical(Y)

            # shuffle X and Y
            X_data, Y_data = shuffle(X, Y)

            yield X_data, Y_data


def main():
    # NOTE: Pulling up the N-JESTER (reduced) dataset
    dataset_class_path = 'D:\LowPowerActionRecognition\CNN\JESTER\data'
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
    # data.load_JESTER_features('train_10_class')

    # NOTE: uncomment below for feature sequence loading testing
    # X_test, y_test = data.load_JESTER_sequences('train')
    #
    # print('shape for x_test:', X_test.shape)
    # print('shape for y_test:', y_test.shape)
    #
    # NOTE: uncomment below for load generator testing
    # generator = data.load_generator('test_10_class')
    # num = 0
    # for i in generator:
    #     num = num + 1
    #     print(num)

if __name__ == '__main__':
    main()
