from .Layer import Layer
import numpy as np

class SimpleNeuralNetwork(Layer):
    def __init__(self, input_size, output_size, activation_function, activation_function_d, output_function):
        super().__init__()
        self.parameters['W1'] = np.random.randn(output_size, input_size) * 0.01
        self.parameters['b1'] = np.zeros((output_size, 1))
        self.parameters['W2'] = np.random.randn(output_size, output_size) * 0.01
        self.parameters['b2'] = np.zeros((output_size, 1))
        self.activation_function = activation_function
        self.activation_function_d = activation_function_d
        self.output_function = output_function

    def forward(self, X):
        # Almacenamos activaciones intermedias para retropropagación
        self.Z1 = self.parameters['W1'].dot(X) + self.parameters['b1']
        self.A1 = self.activation_function(self.Z1)
        self.Z2 = self.parameters['W2'].dot(self.A1) + self.parameters['b2']
        self.A2 = self.output_function(self.Z2)

    def backward(self, X, Y):
        m = Y.size
        dZ2 = self.A2.copy()
        dZ2[Y, np.arange(m)] -= 1
        dZ2 /= m

        # Guardamos los gradientes para usar en el optimizador
        self.gradients = {}
        self.gradients['W2'] = dZ2.dot(self.A1.T)
        self.gradients['b2'] = np.sum(dZ2, axis=1, keepdims=True)
        
        dZ1 = self.parameters['W2'].T.dot(dZ2) * self.activation_function_d(self.Z1)
        self.gradients['W1'] = dZ1.dot(X.T)
        self.gradients['b1'] = np.sum(dZ1, axis=1, keepdims=True)