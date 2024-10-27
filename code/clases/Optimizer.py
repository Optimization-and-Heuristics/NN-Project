class SGDOptimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, model):
        for key, param in model.parameters.items():
            model.parameters[key] -= self.learning_rate * model.gradients[key]