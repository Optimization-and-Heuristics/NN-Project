class Layer:
    def __init__(self):
        self.parameters = {}

    def forward(self, X):
        raise NotImplementedError("Debe implementarse en una subclase específica.")

    def backward(self, grad_output):
        raise NotImplementedError("Debe implementarse en una subclase específica.")