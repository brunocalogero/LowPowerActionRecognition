# Copyright: (c) 2019, Bruno Calogero <brunocalogero@hotmail.com>
# GNU General Public License v3.0+ (see COPYING or https://www.gnu.org/licenses/gpl-3.0.txt)

"""
Train our RNN on extracted features or images.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import ResearchModels
from data_import import Dataset
import time
import keras
import os.path
import datetime as dt
import tensorflow as tf

# Setting up GPU / CPU, set log_device_placement to True to see what uses GPU and what uses CPU
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, device_count = {'GPU': 1 , 'CPU': 1})
sess = tf.Session(config=config)
keras.backend.set_session(sess)


def train(data_type, seq_length, model, class_path, saved_model=None,
          class_limit=None, image_shape=None,
          features=False, batch_size=32, nb_epoch=100, num_classes=4):
    # Helper: Save the model.
    checkpointer = ModelCheckpoint(
        filepath=os.path.join(class_path, 'CNN', 'JESTER', 'scripts', 'multiple_frames', 'checkpoints', model + '-' + data_type + '-' + '10_class' + \
            '.{epoch:03d}-{val_loss:.3f}.hdf5'),
        verbose=1,
        save_best_only=True)

    # Helper: TensorBoard
    tb = TensorBoard(log_dir=os.path.join(class_path, 'CNN', 'JESTER', 'scripts', 'multiple_frames', 'tf_logs', model))

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=5)

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join(class_path, 'CNN', 'JESTER', 'scripts', 'multiple_frames', 'result_logs', model + '-' + 'training-' + '-10_class-' +\
        str(timestamp) + '.log'))


    dataset_class_path = '{0}/CNN/JESTER/data'.format(class_path)
    data = Dataset(path=dataset_class_path)

    if features:
        # get sequence feature data (post InceptionV3 with imagenet)
        start_time = dt.datetime.now()
        print('Start sequence data import {}'.format(str(start_time)))

        X_train, y_train = data.load_JESTER_sequences('train_10_class', categorical=True)
        X_test, y_test = data.load_JESTER_sequences('test_10_class', categorical=True)

        end_time = dt.datetime.now()
        print('Stop load sequence data time {}'.format(str(end_time)))

        elapsed_time= end_time - start_time
        print('Elapsed load sequence data time {}'.format(str(elapsed_time)))

    elif features == None:

        start_time = dt.datetime.now()
        print('Start sequence data import {}'.format(str(start_time)))

        X_train, y_train = data.load_JESTER('train_10_class', categorical=True)
        X_test, y_test = data.load_JESTER('test_10_class', categorical=True)

        end_time = dt.datetime.now()
        print('Stop load sequence data time {}'.format(str(end_time)))

        elapsed_time= end_time - start_time
        print('Elapsed load sequence data time {}'.format(str(elapsed_time)))

    else:

        start_time = dt.datetime.now()
        print('Start data import {}'.format(str(start_time)))

        generator = data.load_generator('test_10_class')
        test_generator = data.load_generator('test_10_class')

        end_time = dt.datetime.now()
        print('Stop load data time {}'.format(str(end_time)))

        elapsed_time= end_time - start_time
        print('Elapsed load data time {}'.format(str(elapsed_time)))

    # Get the model.
    rm = ResearchModels(num_classes, model, seq_length, saved_model)

    # Fit!
    if features or features == None:
        # used for LSTM (feauters loaded after InceptionV3)
        start_time = dt.datetime.now()
        print('Start sequence train data fit {}'.format(str(start_time)))

        rm.model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1,
            callbacks=[tb, early_stopper, csv_logger, checkpointer],
            epochs=nb_epoch)

        end_time = dt.datetime.now()
        print('Stop sequence train data fit {}'.format(str(end_time)))

        elapsed_time= end_time - start_time
        print('Elapsed sequence train data fitting time {}'.format(str(elapsed_time)))
    else:
        # Use standard fit (all other research models)
        start_time = dt.datetime.now()
        print('Start train data fit {}'.format(str(start_time)))

        rm.model.fit_generator(
            generator=generator,
            steps_per_epoch=500, # ~ 16725/32 = 522
            epochs=nb_epoch,
            verbose=1,
            callbacks=[tb, early_stopper, csv_logger, checkpointer],
            validation_data=test_generator,
            validation_steps=50, # ~ 2008/32 = 62.75
            workers=4)

        end_time = dt.datetime.now()
        print('Stop train data fit {}'.format(str(end_time)))

        elapsed_time= end_time - start_time
        print('Elapsed train data fitting time {}'.format(str(elapsed_time)))


def main():
    """These are the main training settings. Set each before running
    this file."""

    class_path = 'D:/LowPowerActionRecognition'

    # model can be one of lstm, lrcn, mlp, conv_3d, c3d
    model = 'mlp'
    saved_model = None  # None or weights file
    class_limit = None
    seq_length = 12
    features = True  # set to true if using lstm or mlp
    batch_size = 32
    nb_epoch = 50
    num_classes = 10 # change if more or less classes

    # Chose images or features and image shape based on network.
    if model in ['conv_3d', 'c3d', 'lrcn']:
        data_type = 'images'
        image_shape = (100, 176, 2)
    elif model in ['lstm', 'mlp']:
        data_type = 'features'
        image_shape = None
    else:
        raise ValueError("Invalid model. See train.py for options.")

    train(data_type, seq_length, model, class_path=class_path, saved_model=saved_model,
          class_limit=class_limit, image_shape=image_shape,
          features=features, batch_size=batch_size, nb_epoch=nb_epoch, num_classes=num_classes)

if __name__ == '__main__':
    main()
