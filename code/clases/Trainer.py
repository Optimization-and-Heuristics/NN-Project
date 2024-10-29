from .Test import *
import time

class Trainer:
    def __init__(self, nn, optimizer, loss_function, loss_function_grad):
        self.nn = nn
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.loss_function_grad = loss_function_grad

    def train(self, X_train, Y_train, X_val, Y_val, epochs=100, print_every=10):
        print("-" * 68)
        loss = []
        acc = []
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            
            predictions = self.nn.forward(X_train)
            loss.append(self.loss_function(Y_train, predictions))
            acc.append(evaluate(predictions, Y_train))
            grad_output = self.loss_function_grad(Y_train, predictions)
            self.nn.backward(grad_output)
            for layer in self.nn.layers:
                self.optimizer.update(layer, {"weights": layer.grad_weights, "biases": layer.grad_biases}, 
                                      t=epoch)
            
            if epoch % print_every == 0:
                val_pred = predict(self.nn,X_val)
                val_acc = evaluate(val_pred, Y_val)
                val_loss = self.loss_function(Y_val, self.nn.forward(X_val))
                
                print("| Epoch {:3.0f} | time: {:5.2f}s | val loss {:2.3f} | valid accuracy {:2.3f} |".format(
                    epoch, time.time() - epoch_start_time, val_loss, val_acc))
                print("-" * 68)
        
        return loss, acc