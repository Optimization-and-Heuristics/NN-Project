import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def predict(nn, X):
    return nn.forward(X)


def evaluate(predictions, Y):
    return np.mean(np.argmax(predictions, axis=1) == np.argmax(Y, axis=1))


def confusion_matrix(Y_test, Y_pred, class_labels=None, incline=False):
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

    if incline:
        plt.xticks(ticks=np.arange(num_classes), labels=tick_labels, rotation=45)

    # Añadir los valores dentro de cada celda con color de texto dinámico
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, conf_matrix[i, j],
                     ha="center", va="center",
                     color="white" if conf_matrix[i, j] > np.max(conf_matrix) / 2 else "black")

    plt.show()


def plot_training_history_with_validation(train_info, val_info):

    train_epochs = range(1, len(train_info["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(train_epochs, train_info["loss"], label="Training", color="blue")
    ax1.plot(val_info["epoch"], val_info["loss"], label="Validation", color="red")
    ax1.set_title("Loss for training and validation subsets")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(train_epochs, train_info["acc"], label="Training", color="blue")
    ax2.plot(val_info["epoch"], val_info["acc"], label="Validation", color="red")
    ax2.set_title("Accuracy for training and validation subsets")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0.0, 1.0)
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def calculate_metrics(predictions: np.array, Y: np.array, class_labels=None):
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(Y, axis=1)

    classes = np.unique(true_labels)
    metrics = {}
    total_support = len(true_labels)

    weighted_precision_sum = 0
    weighted_recall_sum = 0
    weighted_f1_sum = 0

    for cls in classes:
        tp = np.sum((predicted_labels == cls) & (true_labels == cls))
        fp = np.sum((predicted_labels == cls) & (true_labels != cls))
        fn = np.sum((predicted_labels != cls) & (true_labels == cls))

        support = np.sum(true_labels == cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        if class_labels:
            metrics[class_labels[cls]] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "support": support,
            }
        else:
            metrics[cls] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "support": support,
            }

        weighted_precision_sum += precision * support
        weighted_recall_sum += recall * support
        weighted_f1_sum += f1_score * support

    weighted_precision = weighted_precision_sum / total_support
    weighted_recall = weighted_recall_sum / total_support
    weighted_f1_score = weighted_f1_sum / total_support

    accuracy = np.mean(predicted_labels == true_labels)

    metrics["weighted_avg"] = {
        "precision": weighted_precision,
        "recall": weighted_recall,
        "f1_score": weighted_f1_score,
        "support": total_support,
    }

    return accuracy, metrics


def print_metrics(accuracy, metrics):
    print("Accuracy:", accuracy)
    metrics_df = pd.DataFrame.from_dict(metrics, orient="index")
    metrics_df.index.name = "class"
    print(metrics_df)
