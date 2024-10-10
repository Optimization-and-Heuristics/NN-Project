import numpy as np

class Tools:
    def __init__(self):
        pass

    @staticmethod
    def one_hot_encoding(Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y
