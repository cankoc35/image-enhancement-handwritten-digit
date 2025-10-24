import tensorflow as tf

def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Rescaling(1./255),

        tf.keras.layers.Conv2D(filters=8, kernel_size=5, strides=1,
                               padding='same', activation='relu',
                               kernel_initializer=tf.keras.initializers.VarianceScaling()),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),

        tf.keras.layers.Conv2D(filters=16, kernel_size=5, strides=1,
                               padding='same', activation='relu',
                               kernel_initializer=tf.keras.initializers.VarianceScaling()),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
