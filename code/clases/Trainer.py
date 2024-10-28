import numpy as np
import time  # Importar la librería `time`

class Trainer:
    def __init__(self, model, optimizer, loss_function):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function

    def train(self, X_train, Y_train, X_val, Y_val, epochs=100, print_every=10):
        print("-" * 68)
        for epoch in range(epochs + 1):
            val_start_time = time.time()
            
            # Entrenamiento en el conjunto de entrenamiento
            self.model.forward(X_train)
            self.model.backward(X_train, Y_train)
            self.optimizer.update(self.model)

            # Validación en el conjunto de validación cada 10 épocas
            if epoch % print_every == 0:
                val_predictions = self.model.predict(X_val)
                val_accuracy = self.evaluate(val_predictions, Y_val)
                val_loss = self.loss_function(Y_val, self.model.A2)
                print("| Epoch {:3d} | time: {:5.2f}s | val loss {:2.3f} | valid accuracy {:2.3f} |".format(
                    epoch, time.time() - val_start_time, val_loss, val_accuracy))
                print("-" * 68)
                val_start_time = time.time()

    def evaluate(self, predictions, Y):
        # Calcular el porcentaje de precisión
        return np.mean(predictions == Y)