class Optimizer:
    def __init__(self):
        pass

    def update(self, model):
        raise NotImplementedError("Debe implementarse en una subclase específica.")


class SGDOptimizer(Optimizer):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, model):
        for key, param in model.parameters.items():
            model.parameters[key] -= self.learning_rate * model.gradients[key]


class AdamOptimizer(Optimizer):
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, model):
        self.t += 1
        for key, param in model.parameters.items():
            if key not in self.m:
                self.m[key] = 0
                self.v[key] = 0

            gradient_t = model.gradients[key]

            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * gradient_t
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (gradient_t ** 2)

            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            model.parameters[key] -= self.learning_rate * m_hat / (v_hat ** 0.5 + self.epsilon)