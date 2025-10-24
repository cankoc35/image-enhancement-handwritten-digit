# train.py
import os, datetime
import tensorflow as tf
from data import load_mnist_with_channels
from model_def import build_model

MODEL_PATH = "saved_models/mnist_cnn_model.keras"

def main():
    (x_train, y_train), (x_test, y_test) = load_mnist_with_channels()

    model = build_model()
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy']
    )

    os.makedirs("saved_models", exist_ok=True)
    log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(
        x_train, y_train,
        epochs=10,
        validation_data=(x_test, y_test),
        callbacks=[tb]
    )

    print("\nEvaluating…")
    tr_loss, tr_acc = model.evaluate(x_train, y_train, verbose=2)
    te_loss, te_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Train acc: {tr_acc:.4f}  loss: {tr_loss:.4f}")
    print(f"Test  acc: {te_acc:.4f}  loss: {te_loss:.4f}")

    model.save(MODEL_PATH)   # modern .keras format
    print(f"Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    main()
