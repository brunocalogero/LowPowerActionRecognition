'''
make sure to run this in containing forlder (sys)
'''
import sys
sys.path.append('../')

import os.path
import threading
import random

import numpy as np
import datetime as dt
from sklearn.utils import shuffle
from keras.utils import to_categorical
from CNN.JESTER.scripts.multiple_frames.data_import import Dataset
from keras.models import load_model

numy = 0

def save_single_example(example, label, train_test):

    global numy

    if label == 0:
        numy = numy + 1
        np.save('./data/n_{0}/Swiping_Down/{1};Swiping_Down_decoded.npy'.format(train_test, numy), example)
    elif label == 1:
        numy = numy + 1
        np.save('./data/n_{0}/Swiping_Up/{1};Swiping_Up_decoded.npy'.format(train_test, numy), example)
    elif label == 2:
        numy = numy + 1
        np.save('./data/n_{0}/Swiping_Left/{1};Swiping_Left_decoded.npy'.format(train_test, numy), example)
    elif label == 3:
        numy = numy + 1
        np.save('./data/n_{0}/Swiping_Right/{1};Swiping_Right_decoded.npy'.format(train_test, numy), example)


def generate_decoded_data(encoder, decoder, generator, train_test):
    # generator returns tuple: ( (32, 12, 100, 176, 2) , (32, 4) )
    for num, (batch, labels) in enumerate(generator):
        # example: (12, 100, 176, 2)
        for example, label in zip(batch, labels):
            decoded_example = []
            # frame: (100, 176, 2)
            for frame in example:
                # pre-processing on frame (padding): (104, 176, 2)
                frame = np.pad(frame, ((2, 2), (0, 0), (0, 0)), 'edge')
                # pre-processing (to float)
                frame_float = (frame.astype('float16'))
                #  pre-processing (normalization)
                frame_float[:] = [x / 5 for x in frame_float]
                # pre-processing add fourth dimension: (1, 104, 176, 2)
                frame_final = frame_float.reshape(1, 104, 176, 2)

                # encode/decode data
                _,_,z2 = encoder.predict(frame_final, batch_size=1)
                data_output = decoder.predict(z2)
                # append to re-obtain (12, 104, 176, 2)
                decoded_example.append(data_output)

            decoded_example = np.array(decoded_example)
            decoded_example = decoded_example.reshape(12, 104, 176, 2)
            # de-normalization and reshaping to (12, 100, 176, 2)
            decoded_example[:] = [x * 5 for x in decoded_example]
            decoded_example = decoded_example[:,1:-3]

            save_single_example(decoded_example, label, train_test)

        if train_test == 'test':
            # 2008/32 = 62.75
            if num == 60:
                break
        elif train_test == 'train':
            # ~ 16725/32 = 522
            if num == 520:
                break

def main():

    # NOTE: user selects flag type
    train_test = 'train'

    dataset_class_path = 'D:\LowPowerActionRecognition\CNN\JESTER\data'

    data = Dataset(path=dataset_class_path)

    generator = data.load_generator(train_test, categorical=False, regeneration=False)

    # uncomment for sanity check of 32 batch-yielding generator
    # num = 0
    # for i in generator:
    #     print(i[0].shape)

    encoder = load_model('./models/encoder_Chingis_4.h5', custom_objects={'batch_size': 1,'latent_dim': 30,'epsilon_std':0.001})
    decoder = load_model('./models/decoder_Chingis_4.h5', custom_objects={'batch_size': 1,'latent_dim': 30,'epsilon_std':0.001})

    # encoder.summary()
    # decoder.summary()

    generate_decoded_data(encoder, decoder, generator, train_test)

if __name__ == '__main__':
    main()
