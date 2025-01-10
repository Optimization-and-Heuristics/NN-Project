from .Test import *
import time
import numpy as np

class Trainer:
    def __init__(self, nn, optimizer, loss_function, loss_function_grad, 
                 evaluate_fn=None, batch_size=32):
        """
        nn: La red neuronal.
        optimizer: El optimizador para actualizar los parámetros.
        loss_function: Función de pérdida (ej. MSE o Cross-Entropy).
        loss_function_grad: Gradiente de la función de pérdida.
        evaluate_fn: Función de evaluación personalizada (ej. MSE, Accuracy).
        batch_size: Tamaño del batch (default=32).
        """
        self.nn = nn
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.loss_function_grad = loss_function_grad
        self.evaluate_fn = evaluate_fn  # Métrica personalizada
        self.batch_size = batch_size

    def train(self, X_train, Y_train, X_val, Y_val, 
              epochs=100, print_every=10,
              early_stopping=False, patience=5, min_delta=1e-4):
        """
        Entrena la red neuronal con posibilidad de Early Stopping.
        
        Parámetros:
        -----------
        X_train, Y_train: Datos de entrenamiento.
        X_val, Y_val: Datos de validación.
        epochs: Número máximo de épocas para entrenar.
        print_every: Frecuencia con la que se mostrará el estado del entrenamiento.
        early_stopping: Activa/Desactiva la funcionalidad de Early Stopping.
        patience: Número de épocas a esperar antes de detener si no hay mejora.
        min_delta: Diferencia mínima requerida para considerar que hubo mejora.
        """

        m = X_train.shape[0]
        loss = []
        metrics = []
        val_info = {"epoch": [], "loss": [], "metric": []}

        # Variables para Early Stopping
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()

            # Mezclamos los datos en cada época
            indices = np.arange(m)
            np.random.shuffle(indices)
            X_train = X_train[indices]
            Y_train = Y_train[indices]

            batch_losses = []

            # Entrenamiento en batches
            for start in range(0, m, self.batch_size):
                end = start + self.batch_size
                X_batch = X_train[start:end]
                Y_batch = Y_train[start:end]

                # Forward
                predictions = self.nn.forward(X_batch)

                # Cálculo de la pérdida
                batch_loss = self.loss_function(Y_batch, predictions)
                batch_losses.append(batch_loss)

                # Backward
                grad_output = self.loss_function_grad(Y_batch, predictions)
                self.nn.backward(grad_output)

                # Actualización de parámetros
                for layer in self.nn.layers:
                    self.optimizer.update(
                        layer, 
                        {"weights": layer.grad_weights, "biases": layer.grad_biases},
                        t=epoch
                    )

            # Pérdida media de la época
            epoch_loss = np.mean(batch_losses)
            loss.append(epoch_loss)

            # Métrica (opcional) de entrenamiento en el último batch
            if self.evaluate_fn:
                metrics.append(self.evaluate_fn(predictions, Y_batch))

            # Evaluación en validación
            val_pred = self.nn.forward(X_val)
            val_loss = self.loss_function(Y_val, val_pred)
            val_metric = self.evaluate_fn(val_pred, Y_val)

            # Impresión de resultados
            if epoch % print_every == 0 or epoch == 1:
                if self.evaluate_fn == mse_evaluate:
                    residuals = Y_val - val_pred
                    residual_mean = np.mean(residuals)
                    print("| Epoch {:3.0f} | time: {:5.2f}s | val loss (MSE) {:2.3f} | Residual Mean {:2.3f} |".format(
                        epoch, time.time() - epoch_start_time, val_loss, residual_mean))
                else:
                    print("| Epoch {:3.0f} | time: {:5.2f}s | val loss {:2.3f} | val metric {:2.3f} |".format(
                        epoch, time.time() - epoch_start_time, val_loss, val_metric))

            # Guardar información de validación
            val_info["metric"].append(val_metric)
            val_info["epoch"].append(epoch)
            val_info["loss"].append(val_loss)

            # -------------------------
            # Early Stopping
            # -------------------------
            # Verificamos si hay mejora significativa en val_loss
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            # Si se activa Early Stopping y se supera la paciencia, detener
            if early_stopping and patience_counter >= patience:
                print("Early stopping activado en la época {:3.0f}".format(epoch))
                break

        return loss, metrics, val_info
