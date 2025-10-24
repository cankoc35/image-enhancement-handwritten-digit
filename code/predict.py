# predict.py
import numpy as np
import matplotlib.pyplot as plt
import math
from tensorflow import keras
from data import load_mnist_with_channels

MODEL_PATH = "saved_models/mnist_cnn_model.keras"

def main():
    (_, _), (x_test, y_test) = load_mnist_with_channels()
    model = keras.models.load_model(MODEL_PATH)

    # get predictions
    predictions = model.predict(x_test, verbose=0)
    pred_labels = np.argmax(predictions, axis=1)

    # overall accuracy
    acc = np.mean(pred_labels == y_test)
    print(f"Test accuracy: {acc:.4f}")

    # ==== Visualization ====
    # numbers_to_display = 100  # number of test images to display
    # num_cells = math.ceil(math.sqrt(numbers_to_display))

    # plt.figure(figsize=(15, 15))

    # for i in range(numbers_to_display):
    #     predicted = pred_labels[i]
    #     actual = y_test[i]

    #     color_map = 'Greens' if predicted == actual else 'Reds'
    #     plt.subplot(num_cells, num_cells, i + 1)
    #     plt.imshow(x_test[i].reshape(28, 28), cmap=color_map)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.xlabel(f"{predicted}", fontsize=9)
    #     plt.grid(False)

    # plt.subplots_adjust(hspace=0.5, wspace=0.5)
    # plt.suptitle("Model Predictions (Green = Correct, Red = Wrong)", fontsize=16)
    # plt.show()

if __name__ == "__main__":
    main()
