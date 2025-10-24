import tensorflow as tf

def load_mnist_with_channels():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # add channel dim (28,28) -> (28,28,1)
    x_train = x_train[..., None]
    x_test  = x_test[..., None]
    return (x_train, y_train), (x_test, y_test)
