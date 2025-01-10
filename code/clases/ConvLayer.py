from .Layer import Layer
import numpy as np
from scipy import signal


def correlate2d(inputs: np.array, kernels, mode):
    return signal.correlate2d(inputs, kernels, mode)


class ConvLayer(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, X):
        # X.shape = (batch_size, input_depth, input_height, input_width)
        batch_size = X.shape[0]
        self.input = X
        self.output = np.zeros((batch_size, *self.output_shape))

        for b in range(batch_size):
            for i in range(self.depth):
                for j in range(self.input_depth):
                    self.output[b, i] += correlate2d(self.input[b, j].reshape(self.input_shape[1], self.input_shape[2]), self.kernels[i, j], "valid")
                self.output[b, i] += self.biases[i]
        return self.output

    def backward(self, grad_output, learning_rate=0.04):
        # grad_output.shape = (batch_size, depth, output_height, output_width)
        batch_size = grad_output.shape[0]
        self.kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros((batch_size, *self.input_shape))
        self.biases_gradient = np.zeros_like(self.biases)

        for b in range(batch_size):
            for i in range(self.depth):
                for j in range(self.input_depth):
                    self.kernels_gradient[i,j] = correlate2d(self.input[b, j], grad_output[b, i], "valid")
                    input_gradient[b, j] += correlate2d(grad_output[b, i], self.kernels[i,j], "full")
                self.biases_gradient[i] += grad_output[b, i]

        self.kernels_gradient /= batch_size
        self.biases_gradient /= batch_size

        self.kernels -= learning_rate * self.kernels_gradient
        self.biases -= learning_rate * self.biases_gradient
        return input_gradient


if __name__ == '__main__':
    x = [[[1, 2], [3, 4]]]
    lay = ConvLayer((1,2,2), 2, 1)
    aim = [[[0]]]
    for e in range(50):
        y = lay.forward(x)
        print(f'Epoch {e} - {y}')
        lay.backward(-(aim-y)**2, 0.001)
