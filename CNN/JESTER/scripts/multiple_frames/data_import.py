import os
import numpy as np
import datetime as dt
from sklearn.utils import shuffle
from keras.utils import to_categorical



class Dataset():
    '''
    Pull the neuromorphic JESTER dataset with flexibility and provide additional processing functionality
    '''

    def __init__(self, path, num_train=-1, num_test=-1, image_shape=(100, 176, 2)):

        self.image_shape = image_shape
        self.num_train = num_train
        self.num_test = num_test
        self.path = path


    def load_JESTER(self, train_test, categorical=False):
        """
        Imports the neuromorphic JESTER Dataset
        """

        xs = []
        ys = []

        labels = ['Swiping_Down', 'Swiping_Left', 'Swiping_Right', 'Swiping_Up']

        data = 'train' if train_test == 'train' else 'test'

        for label in labels:
            # print('{0}/n_{1}/{2}'.format(self.path, data, label))
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


def main():
    # NOTE: Pulling up the N-JESTER (reduced) dataset
    dataset_class_path = '/Users/brunocalogero/Desktop/LowPowerActionRecognition/CNN/JESTER/data'
    data = Dataset(path=dataset_class_path)

    start_time = dt.datetime.now()
    print('Start data import {}'.format(str(start_time)))

    X_test, Y_test = data.load_JESTER('test', categorical=True)
    print('X_test of shape:', X_test.shape)
    print('Y_test of shape:', Y_test.shape)


    X_train, Y_train = data.load_JESTER('train', categorical=True)
    print('X_train of shape:', X_train.shape)
    print('Y_train of shape:', Y_train.shape)

    end_time = dt.datetime.now()
    print('Stop load data time {}'.format(str(end_time)))

    elapsed_time= end_time - start_time
    print('Elapsed load data time {}'.format(str(elapsed_time)))

    # sanity check
    print('Sanity Check for shuffling:')
    print(Y_test[4])
    print(Y_test[5])
    print(Y_test[6])


if __name__ == '__main__':
    main()
