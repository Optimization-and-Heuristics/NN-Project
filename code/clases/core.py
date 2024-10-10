from clases.activation_function import ActivationFunction
from clases.tools import Tools
import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, output_size, alpha):
        self.W1 = np.random.normal(size=(output_size, input_size)) * 0.01 
        self.b1 = np.zeros((output_size, 1))
        self.W2 = np.random.normal(size=(output_size, output_size)) * 0.01 
        self.b2 = np.zeros((output_size, 1))
        self.alpha = alpha
        
    def forward(self, input):
        self.Z1 = self.W1.dot(input) + self.b1
        self.A1 = ActivationFunction().relu(self.Z1)
        self.Z2 = self.W2.dot(self.A1) + self.b2
        self.A2 = ActivationFunction().softmax(self.Z2)
    
    def backward(self, X, Y):
        one_hot_Y = Tools().one_hot_encoding(Y) 
        self.dZ2 = (self.A2 - one_hot_Y) / Y.size
        self.dW2 = self.dZ2.dot(self.A1.T)
        self.db2 = np.sum(self.dZ2)
        self.dZ1 = self.W2.T.dot(self.dZ2) * ActivationFunction().relu_derivative(self.Z1)
        self.dW1 = self.dZ1.dot(X.T)
        self.db1 = np.sum(self.dZ1)

    def update_params(self):
        self.W1 = self.W1 - self.alpha * self.dW1
        self.b1 = self.b1 - self.alpha * self.db1
        self.W2 = self.W2 - self.alpha * self.dW2
        self.b2 = self.b2 - self.alpha * self.db2
    
    def predict(self):
        return np.argmax(self.A2, axis=0)

    def evaluate(self, predictions, Y):
        print(predictions)
        return np.sum(predictions == Y) / Y.size

    def train(self, X, Y, epochs):
        for i in range(epochs):
            self.forward(X)
            self.backward(X, Y)
            self.update_params()
            if i % 10 == 0:
                predictions = self.predict()
                print(f'Iteration: {i}')
                print(f'Accuracy: {self.evaluate(predictions, Y):.2f}')
        return self.W1, self.b1, self.W2, self.b2
        