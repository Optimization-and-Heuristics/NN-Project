import numpy as np
import matplotlib.pyplot as plt

def confusion_matrix(Y_test, Y_pred, class_labels=None, inclain=False):
    # Número de clases (suponemos que sabemos la cantidad de clases)
    num_classes = np.max([Y_test.max(), Y_pred.max()]) + 1

    # Crear la matriz de confusión manualmente
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(Y_test, Y_pred):
        conf_matrix[true_label, pred_label] += 1

    # Visualizar la matriz de confusión
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, cmap="Blues")
    plt.title("Confusion Matrix")
    
    # Añadir etiquetas si se proporcionan
    tick_labels = class_labels if class_labels is not None else range(num_classes)
    plt.xticks(ticks=np.arange(num_classes), labels=tick_labels)
    plt.yticks(ticks=np.arange(num_classes), labels=tick_labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    
    if inclain:
        # Inclinar los ticks del eje x
        plt.xticks(ticks=np.arange(num_classes), labels=tick_labels, rotation=45)

    # Añadir anotaciones en cada celda
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, conf_matrix[i, j], ha="center", va="center", color="black")

    plt.show()

def predict(X, model):
    # Forward pass para obtener las predicciones
    model.forward(X)
    return np.argmax(model.A2, axis=0)