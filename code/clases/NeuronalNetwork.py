from .DenseLayer import DenseLayer

class NeuronalNetwork:
    def __init__(self, input_size, layers_config):
        self.layers = []
        in_size = input_size

        # Crear capas de acuerdo a layers_config
        for layer in layers_config:
            output_size = layer['output_size']
            activation = layer['activation']
            self.layers.append(DenseLayer(input_size=in_size, output_size=output_size, activation=activation))
            in_size = output_size  # La salida de la capa anterior es la entrada de la siguiente

    def forward(self, X):
        for layer in self.layers:
            x = layer.forward(X)
        return x

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)