from .Layer import Layer
import numpy as np


class FlattenLayer(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, X):
        batch_size = X.shape[0]
        return np.reshape(X, (batch_size, -1))

    def backward(self, grad_output):
        batch_size = grad_output.shape[0]
        return np.reshape(grad_output, (batch_size, *self.input_shape))
