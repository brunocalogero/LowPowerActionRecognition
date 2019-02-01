import numpy as np
import os
path='D:/JESTER/n_JESTER'

def load_JESTER(path):
    """
    Imports the JESTER Dataset (12,2,100,176)
    """
    xs_train = []
    ys_train = []
    xs_test = []
    ys_test = []

    classList=['Swiping_Left', 'Swiping_Right', 'Swiping_Up', 'Swiping_Down']
    for class_index in classList:
        for (root, dirs, f) in os.walk('{0}/n_Train/{1}'.format(path, str(class_index))):
            cnt=0
            print('Loading Train set')
            for file in f:
                cnt += 1
                if (cnt % 100 == 0):
                    print('Percent loaded: {}%'.format(round((cnt / len(f)) * 100, 2)))
                X=np.load('{0}/n_Train/{1}/{2}'.format(path, str(class_index), file))
                xs_train.append(X)
                ys_train.append(class_index)

        for (root, dirs, f) in os.walk('{0}/n_Test/{1}'.format(path, str(class_index))):
            cnt=0
            print('Loading Test set')
            for file in f:
                cnt += 1
                if (cnt % 100 == 0):
                    print('Percent loaded: {}%'.format(round((cnt / len(f)) * 100, 2)))
                X=np.load('{0}/n_Test/{1}/{2}'.format(path, str(class_index), file))
                xs_test.append(X)
                ys_test.append(class_index)

    Xtr = np.array(xs_train)
    Ytr = np.array(ys_train)
    Xte = np.array(xs_test)
    Yte = np.array(ys_test)

    return Xtr, Ytr, Xte, Yte


Xtr, Ytr, Xte, Yte=load_JESTER(path)
