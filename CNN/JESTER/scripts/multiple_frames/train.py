"""
Train our RNN on extracted features or images.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import ResearchModels
from data_import import Dataset
import time
import datetime as dt
import os.path


# Setting up GPU / CPU, set log_device_placement to True to see what uses GPU and what uses CPU
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, device_count = {'GPU': 1 , 'CPU': 1})
sess = tf.Session(config=config)
keras.backend.set_session(sess)


def train(data_type, seq_length, model, class_path, saved_model=None,
          class_limit=None, image_shape=None,
          load_to_memory=False, batch_size=32, nb_epoch=100, num_classes=4):
    # Helper: Save the model.
    checkpointer = ModelCheckpoint(
        filepath=os.path.join(class_path, 'CNN', 'JESTER', 'scripts', 'multiple_frames', 'checkpoints', model + '-' + data_type + \
            '.{epoch:03d}-{val_loss:.3f}.hdf5'),
        verbose=1,
        save_best_only=True)

    # Helper: TensorBoard
    tb = TensorBoard(log_dir=os.path.join(class_path, 'CNN', 'JESTER', 'scripts', 'multiple_frames', 'tf_logs', model))

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=5)

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join(class_path, 'CNN', 'JESTER', 'scripts', 'multiple_frames', 'result_logs', model + '-' + 'training-' + \
        str(timestamp) + '.log'))

    # Get the data and process it.
    # if image_shape is None:
    #     data = DataSet(
    #         seq_length=seq_length,
    #         class_limit=class_limit
    #     )
    # else:
    #     data = DataSet(
    #         seq_length=seq_length,
    #         class_limit=class_limit,
    #         image_shape=image_shape
    #     )
    #
    # # Get samples per epoch.
    # # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
    # steps_per_epoch = (len(data.data) * 0.7) // batch_size
    #
    # if load_to_memory:
    #     # Get data.
    #     X, y = data.get_all_sequences_in_memory('train', data_type)
    #     X_test, y_test = data.get_all_sequences_in_memory('test', data_type)
    # else:
    #     # Get generators.
    #     generator = data.frame_generator(batch_size, 'train', data_type)
    #     val_generator = data.frame_generator(batch_size, 'test', data_type)

    dataset_class_path = '{0}/CNN/JESTER/data'.format(class_path)
    data = Dataset(path=dataset_class_path)

    start_time = dt.datetime.now()
    print('Start data import {}'.format(str(start_time)))

    # X_test, y_test = data.load_JESTER('test', categorical=True)
    X_train, y_train = data.load_JESTER('train', categorical=True)

    end_time = dt.datetime.now()
    print('Stop load data time {}'.format(str(end_time)))

    elapsed_time= end_time - start_time
    print('Elapsed load data time {}'.format(str(elapsed_time)))

    # Get the model.
    rm = ResearchModels(num_classes, model, seq_length, saved_model)

    # Fit!
    start_time = dt.datetime.now()
    print('Start train data fit {}'.format(str(start_time)))

    rm.model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1,
        callbacks=[tb, early_stopper, csv_logger],
        epochs=nb_epoch)

    end_time = dt.datetime.now()
    print('Stop train data fit {}'.format(str(end_time)))

    elapsed_time= end_time - start_time
    print('Elapsed train data fitting time {}'.format(str(elapsed_time)))

    # if load_to_memory:
    #     # Use standard fit.
    #     rm.model.fit(
    #         X,
    #         y,
    #         batch_size=batch_size,
    #         validation_data=(X_test, y_test),
    #         verbose=1,
    #         callbacks=[tb, early_stopper, csv_logger],
    #         epochs=nb_epoch)
    # else:
    #     # Use fit generator.
    #     rm.model.fit_generator(
    #         generator=generator,
    #         steps_per_epoch=steps_per_epoch,
    #         epochs=nb_epoch,
    #         verbose=1,
    #         callbacks=[tb, early_stopper, csv_logger, checkpointer],
    #         validation_data=val_generator,
    #         validation_steps=40,
    #         workers=4)

    # # fit the best alg to the training data
    # start_time = dt.datetime.now()
    # print('Start learning with best params at {}'.format(str(start_time)))
    #
    # model.fit(X_train, to_categorical(Y_train, num_classes=4), validation_data=(X_val, to_categorical(Y_val, num_classes=4)), epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[TensorBoard(log_dir='tf_logs/1/train')])
    #
    # end_time = dt.datetime.now()
    # print('Stop learning {}'.format(str(end_time)))
    # elapsed_time= end_time - start_time
    # print('Elapsed learning time {}'.format(str(elapsed_time)))
    #
    #
    # predictions_scores = model.predict(X_testy, batch_size=batch_size)
    #
    # print(len(predictions_scores) == len(Y_test))
    # print(predictions_scores)
    # print(Y_test)
    #
    #
    # # Actual Inference
    # predictions = model.predict_classes(X_testy, batch_size=batch_size)
    #
    # print(accuracy_score(Y_test, predictions))
    #
    # # Now predict the value of the test
    # expected = Y_test
    #
    # print("Classification report for classifier:\n%s\n", metrics.classification_report(expected, predictions))
    #
    # cm = metrics.confusion_matrix(expected, predictions)
    # print("Confusion matrix:\n%s" % cm)
    #
    # print("Accuracy={}".format(metrics.accuracy_score(expected, predictions)))


def main():
    """These are the main training settings. Set each before running
    this file."""

    class_path = '/Users/brunocalogero/Desktop/LowPowerActionRecognition/'

    # model can be one of lstm, lrcn, mlp, conv_3d, c3d
    model = 'c3d'
    saved_model = None  # None or weights file
    class_limit = None  # int, can be 1-101 or None
    seq_length = 12
    load_to_memory = False  # pre-load the sequences into memory
    batch_size = 32
    nb_epoch = 1
    num_classes = 4

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
          load_to_memory=load_to_memory, batch_size=batch_size, nb_epoch=nb_epoch, num_classes=num_classes)

if __name__ == '__main__':
    main()
