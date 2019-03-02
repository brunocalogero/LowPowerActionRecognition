import os.path
import numpy as np

from keras.models import load_model
from models import ResearchModels


def import_sequence_image(flag):
    """
    """
    if flag == 0:
        data_path = 'D:/LowPowerActionRecognition/CNN/JESTER/data/n_test/Swiping_Down/2999;Swiping_Down.npy'
    elif flag == 1:
        data_path = 'D:/LowPowerActionRecognition/CNN/JESTER/data/n_test/Swiping_Up/691;Swiping_Up.npy'
    elif flag == 2:
        data_path = 'D:/LowPowerActionRecognition/CNN/JESTER/data/n_test/Swiping_Left/2052;Swiping_Left.npy'
    elif flag == 3:
        data_path = 'D:/LowPowerActionRecognition/CNN/JESTER/data/n_test/Swiping_Right/2240;Swiping_Right.npy'

    seq = np.load(data_path)
    seq = seq.reshape(1, 12, 100, 176, 2)

    # sanity check
    print(seq.shape)

    return seq

if __name__ == '__main__':

    num_classes = 4
    seq_length = 12
    model = 'lrcn'

    rm = ResearchModels(num_classes, model, seq_length)

    data = import_sequence_image(3)
    model = load_model('./checkpoints/lrcn-images.013-0.340.hdf5')
    prediction = model.predict(data)

    print(prediction)
