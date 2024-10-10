import numpy as np
from clases.activation_function import ActivationFunction
import matplotlib.pyplot as plt

class TestNN:
    def __init__(self, W1, B1, W2, B2, X, Y):
        self.W1 = W1
        self.b1 = B1
        self.W2 = W2
        self.b2 = B2
        self.X = X
        self.Y = Y

    def forward(self, input):
        self.Z1 = self.W1.dot(input) + self.b1
        self.A1 = ActivationFunction().relu(self.Z1)
        self.Z2 = self.W2.dot(self.A1) + self.b2
        self.A2 = ActivationFunction().softmax(self.Z2)

    def predict(self):
        return np.argmax(self.A2, axis=0)

    def make_predictions(self, X):
        self.forward(X)
        predictions = self.predict()
        return predictions

    def test_prediction(self, index):
        current_image = self.X[:, index, None]
        prediction = self.make_predictions(self.X[:, index, None])
        label = self.Y[index]
        print("Prediction: ", prediction)
        print("Label: ", label)

        current_image = current_image.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(current_image, interpolation='nearest')
        plt.show()