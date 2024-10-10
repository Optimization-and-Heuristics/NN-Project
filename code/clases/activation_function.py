import numpy as np

class ActivationFunction:
    def __init__(self):
        pass

    @staticmethod
    def relu(Z):
        return np.maximum(0, Z)
    
    @staticmethod
    def relu_derivative(Z):
        return Z > 0
    
    @staticmethod
    def softmax(Z):
        Z -= np.max(Z, axis=0)  # Subtract max value for numerical stability
        A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
        return A