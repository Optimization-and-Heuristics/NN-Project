import numpy as np

PARAMS = ['weights', 'biases']

class Optimizer:
    def __init__(self):
        pass

    def update(self, layer, grad, t=None):
        raise NotImplementedError("Debe implementarse en una subclase específica.")


class SGDOptimizer(Optimizer):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, layer, grad, t=None):
        for param in PARAMS:
            setattr(layer, param, getattr(layer, param) - self.learning_rate * grad[param])


class AdamOptimizer(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}

    def update(self, layer, grad, t):
        if layer not in self.m:
            self.m[layer] = {param: np.zeros_like(getattr(layer, param)) for param in PARAMS}
            self.v[layer] = {param: np.zeros_like(getattr(layer, param)) for param in PARAMS}
        
        for param in PARAMS:
            self.m[layer][param] = self.beta1 * self.m[layer][param] + (1 - self.beta1) * grad[param]
            self.v[layer][param] = self.beta2 * self.v[layer][param] + (1 - self.beta2) * (grad[param] ** 2)
            
            m_hat = self.m[layer][param] / (1 - self.beta1 ** t)
            v_hat = self.v[layer][param] / (1 - self.beta2 ** t)
            
            setattr(layer, param, getattr(layer, param) - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon))
        