import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import tensorflow as tf
from tensorflow import keras
import math
from data import load_mnist_with_channels

MODEL_PATH = "saved_models/mnist_cnn_model.keras"

def main():
    # 1️⃣ Load test data & model
    (_, _), (x_test, y_test) = load_mnist_with_channels()
    model = keras.models.load_model(MODEL_PATH)

    # 2️⃣ Predictions
    predictions = model.predict(x_test, verbose=0)
    pred_labels = np.argmax(predictions, axis=1)
    acc = np.mean(pred_labels == y_test)
    print(f"Test accuracy: {acc:.4f}")

    # 3️⃣ Confusion matrix
    cm = tf.math.confusion_matrix(y_test, pred_labels)
    plt.figure(figsize=(9, 7))
    sn.heatmap(cm, annot=True, fmt="d", cmap="Greens", linewidths=.5, square=True)
    plt.title("Confusion Matrix for MNIST Digit Recognition")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

    # 4️⃣ Visualization of predictions (Green = correct, Red = wrong)
    numbers_to_display = 100  # you can change to 49 or 64 if you prefer smaller
    num_cells = math.ceil(math.sqrt(numbers_to_display))

    plt.figure(figsize=(15, 15))
    for i in range(numbers_to_display):
        predicted = pred_labels[i]
        actual = y_test[i]
        color_map = 'Greens' if predicted == actual else 'Reds'
        plt.subplot(num_cells, num_cells, i + 1)
        plt.imshow(x_test[i].reshape(28, 28), cmap=color_map)
        plt.xticks([])
        plt.yticks([])
        label_color = 'green' if predicted == actual else 'red'
        plt.xlabel(f"{predicted}", color=label_color, fontsize=9)
        plt.grid(False)

    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.suptitle("Model Predictions (Green = Correct, Red = Wrong)", fontsize=16)
    plt.show()

if __name__ == "__main__":
    main()
