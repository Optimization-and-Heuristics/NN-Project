import numpy as np

def cross_entropy_loss(Y, A):
    # Cross-entropy loss function
    m = Y.size  # Number of examples
    epsilon = 1e-10  # Small value to avoid log(0)
    log_likelihood = -np.log(A[Y, np.arange(m)] + epsilon)
    loss = np.sum(log_likelihood) / m
    return loss