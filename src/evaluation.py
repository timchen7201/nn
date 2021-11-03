import numpy as np


def accuracy(predict,y):
    return np.mean(np.argmax(predict,axis=1) == np.argmax(y,axis=1))