import numpy as np
import time  # Importar la librería `time`

class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def train(self, X_train, Y_train, X_val, Y_val, epochs=100, print_every=10):
        print("-" * 59)
        for epoch in range(epochs + 1):
            val_start_time = time.time()
            
            # Entrenamiento en el conjunto de entrenamiento
            self.model.forward(X_train)
            self.model.backward(X_train, Y_train)
            self.optimizer.update(self.model)

            # Validación en el conjunto de validación cada 10 épocas
            if epoch % print_every == 0:
                val_predictions = self.predict(X_val)
                val_accuracy = self.evaluate(val_predictions, Y_val)
                print("| end of epoch {:3d} | time: {:5.2f}s | valid accuracy {:8.3f} ".format(
                    epoch, time.time() - val_start_time, val_accuracy))
                print("-" * 59)
                val_start_time = time.time()
    
    def predict(self, X):
        # Forward pass para obtener las predicciones
        self.model.forward(X)
        return np.argmax(self.model.A2, axis=0)

    def evaluate(self, predictions, Y):
        # Calcular el porcentaje de precisión
        return np.mean(predictions == Y)