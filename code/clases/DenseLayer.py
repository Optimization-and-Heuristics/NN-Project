import numpy as np
from .Layer import Layer


class DenseLayer(Layer):
    def __init__(self, input_size, output_size, activation):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros((1, output_size))

        self.activation = activation()

    def forward(self, x):
        # Algorithm 2. Forward Pass para Dense Layer
        self.input = x
        self.output = self.activation.forward(np.dot(x, self.weights) + self.biases)
        return self.output
    
    def backward(self, grad_output):
        # Algorithm 5. Backward Pass para Dense Layer
        grad_output = self.activation.backward(self.output, grad_output)
        self.grad_weights = np.dot(self.input.T, grad_output)
        self.grad_biases = np.sum(grad_output, axis=0, keepdims=True)
        return np.dot(grad_output, self.weights.T)