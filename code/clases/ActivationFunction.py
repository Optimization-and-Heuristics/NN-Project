import numpy as np

class ActivationFunction:
    def forward(self, x):
        raise NotImplementedError

    def backward(self, x, grad_output):
        raise NotImplementedError


class ReLU(ActivationFunction):
    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, x, grad_output):
        return grad_output * (x > 0)


class Softmax(ActivationFunction):
    def forward(self, x):
        exp_values = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=-1, keepdims=True)

    def backward(self, output, grad_output):
        gradients = []
        for i, gradient in enumerate(grad_output):
            jacobian_matrix = np.diagflat(output[i]) - np.dot(output[i][:, None], output[i][None, :])
            gradients.append(np.dot(jacobian_matrix, gradient))
        return np.array(gradients)

class Linear(ActivationFunction):
    def forward(self, x):
        return x

    def backward(self, output, grad_output):
        return grad_output

class Softplus(ActivationFunction):
    def forward(self, x):
        return np.log(1 + np.exp(x))

    def backward(self, x, grad_output):
        return grad_output * (1 / (1 + np.exp(-x)))


class Tanh(ActivationFunction):
    def forward(self, x):
        return np.tanh(x)

    def backward(self, x, grad_output):
        return grad_output * (1 - np.tanh(x) ** 2)


class Sigmoid(ActivationFunction):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, x, grad_output):
        sigmoid_output = 1 / (1 + np.exp(-x))
        return grad_output * sigmoid_output * (1 - sigmoid_output)