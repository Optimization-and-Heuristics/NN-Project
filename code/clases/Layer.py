class Layer:
    def __init__(self):
        self.parameters = {}

    def forward(self, X):
        raise NotImplementedError("Debe implementarse en una subclase específica.")

    def backward(self, X, Y):
        raise NotImplementedError("Debe implementarse en una subclase específica.")