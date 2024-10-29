import numpy as np
import matplotlib.pyplot as plt

def predict(nn,X):
    return nn.forward(X)

def evaluate(predictions, Y):
    return np.mean(np.argmax(predictions, axis=1) == np.argmax(Y, axis=1))

def plot_training_history(loss, acc):
        epochs = range(1, len(loss) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(epochs, loss, label="Training Loss", color="blue")
        ax1.set_title("Training Loss")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(epochs, acc, label="Training Accuracy", color="green")
        ax2.set_title("Training Accuracy")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

def confusion_matrix(Y_test, Y_pred, class_labels=None, inclain=False):
    true_labels = np.argmax(Y_test, axis=1)
    pred_labels = np.argmax(Y_pred, axis=1)
    
    num_classes = Y_test.shape[1]
    
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(true_labels, pred_labels):
        conf_matrix[true_label, pred_label] += 1

    plt.figure(figsize=(12, 6))
    plt.imshow(conf_matrix, cmap="Blues")
    plt.title("Confusion Matrix")
    
    tick_labels = class_labels if class_labels is not None else range(num_classes)
    plt.xticks(ticks=np.arange(num_classes), labels=tick_labels)
    plt.yticks(ticks=np.arange(num_classes), labels=tick_labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    
    if inclain:
        plt.xticks(ticks=np.arange(num_classes), labels=tick_labels, rotation=45)

    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, conf_matrix[i, j], ha="center", va="center", color="black")

    plt.show()