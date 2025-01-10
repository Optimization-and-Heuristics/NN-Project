from .Test import *
import time
import numpy as np
from .DenseLayer import DenseLayer

class Trainer:
    def __init__(self, nn, optimizer, loss_function, loss_function_grad, evaluate_fn=None, batch_size=32):
        """
        nn: La red neuronal.
        optimizer: El optimizador para actualizar los parámetros.
        loss_function: Función de pérdida (ej. MSE o Cross-Entropy).
        loss_function_grad: Gradiente de la función de pérdida.
        evaluate_fn: Función de evaluación personalizada.
        batch_size: Tamaño del batch (default=32).
        """
        self.nn = nn
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.loss_function_grad = loss_function_grad
        self.evaluate_fn = evaluate_fn  # Métrica personalizada (ej. MSE, Accuracy)
        self.batch_size = batch_size  # Tamaño del batch

    def train(self, X_train, Y_train, X_val, Y_val, epochs=100, print_every=10):
        m = X_train.shape[0]
        loss = []
        metrics = []
        val_info = {"epoch": [], "loss": [], "metric": []}

        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()

            indices = np.arange(m)
            np.random.shuffle(indices)
            X_train = X_train[indices]
            Y_train = Y_train[indices]

            batch_losses = []

            for start in range(0, m, self.batch_size):
                end = start + self.batch_size
                X_batch = X_train[start:end]
                Y_batch = Y_train[start:end]

                predictions = self.nn.forward(X_batch)

                batch_loss = self.loss_function(Y_batch, predictions)
                batch_losses.append(batch_loss)
                grad_output = self.loss_function_grad(Y_batch, predictions)

                self.nn.backward(grad_output)
                for layer in self.nn.layers:
                    params = False
                    if type(layer) is DenseLayer:
                        params = {"weights": layer.grad_weights, "biases": layer.grad_biases}
                    if params:
                        self.optimizer.update(
                            layer, params, t=epoch
                        )

            loss.append(np.mean(batch_losses))

            if self.evaluate_fn:
                metrics.append(self.evaluate_fn(predictions, Y_batch))

            if epoch % print_every == 0 or epoch == 1:
                val_pred = self.nn.forward(X_val)
                val_loss = self.loss_function(Y_val, val_pred)

                if self.evaluate_fn == mse_evaluate:
                    residuals = Y_val - val_pred
                    residual_mean = np.mean(residuals)
                    print("| Epoch {:3.0f} | time: {:5.2f}s | val loss (MSE) {:2.3f} | Residual Mean {:2.3f} |".format(
                        epoch, time.time() - epoch_start_time, val_loss, residual_mean))
                else:
                    val_metric = self.evaluate_fn(val_pred, Y_val)
                    print("| Epoch {:3.0f} | time: {:5.2f}s | val loss {:2.3f} | val metric {:2.3f} |".format(
                        epoch, time.time() - epoch_start_time, val_loss, val_metric))
                    val_info["metric"].append(val_metric)

                val_info["epoch"].append(epoch)
                val_info["loss"].append(val_loss)

        return loss, metrics, val_info

