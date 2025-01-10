import numpy as np

def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-10
    # Here we clip the values of y_pred to avoid log(0) and log(1)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    loss = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
    return np.mean(loss)

def cross_entropy_loss_grad(y_true, y_pred):
    # Algorithm 3. Gradiente de la Pérdida de Entropía Cruzada
    epsilon = 1e-10
    # Here we clip the values of y_pred to avoid log(0) and log(1)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    grad = (y_pred - y_true) / (y_pred * (1 - y_pred))
    return grad / y_true.shape[0]

def mean_squared_error(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

def mean_squared_error_grad(y_true, y_pred):
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    return 2 * (y_pred - y_true) / y_true.shape[0]
