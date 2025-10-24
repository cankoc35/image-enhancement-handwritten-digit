import tensorflow as tf
import matplotlib.pyplot as plt
import datetime

# load dataset msnist dataset 
msnist_dataset = tf.keras.datasets.mnist

# x = 28x28px of each grayscale image.
# y = one integer per image (0-9). a single number.
(x_train, y_train), (x_test, y_test) = msnist_dataset.load_data()

# The training dataset consists of 60000 28x28px images.
# The test dataset consists of 10000 28x28px images.
print('x_train:', x_train.shape)
print('y_train:', y_train.shape)
print('x_test:', x_test.shape)
print('y_test:', y_test.shape)

# (60000, 28, 28) → 60000 samples of 28x28px images.
(samples, width, height) = x_train.shape
# MNIST images are grayscale → one channel.
# (Color images would have 3 channels: RGB.)
channels = 1

# Reshape the dataset to add the channels dimension.
# TensorFlow’s conv layers use channels-last by default.
# Color photos would have 3 channels (R,G,B).
x_train_with_channels = x_train.reshape(
    x_train.shape[0], # number of images (samples)
    width, # 28px width
    height, # 28px height
    channels # 1 channel (grayscale)
)

x_test_with_channels = x_test.reshape(
    x_test.shape[0], # number of images (samples)
    width, # 28px width
    height, # 28px height
    channels # 1 channel (grayscale)
)

print('x_train_with_chanels:', x_train_with_channels.shape)
print('x_test_with_chanels:', x_test_with_channels.shape)

# normalize pixel values to the range 0-1.
# Original pixel values are in the range 0-255.
# normalization makes training faster and more stable.
# helps gradient descent move smoothly instead of “jumping” or “stalling.”

# building the model 
# layers(filters) run in order from top to bottom.
# Sequential = a straight line of layers. Data flows layer 1 → layer 2 → ...

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),          # <-- first define the input
    tf.keras.layers.Rescaling(1./255),                 # <-- normalize inside the model
    
    # first layer is a Conv2D layer.
    # Conv2D slides 8 little 5×5 detectors across the image to find edges/strokes.
    tf.keras.layers.Conv2D(
        input_shape=(width, height, channels),
        kernel_size=5,
        filters=8,
        strides=1, # slide the filter one pixel at a time
        padding='same', # output is the same size as input
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling()
    ),
    tf.keras.layers.MaxPool2D(
        pool_size=(2, 2),
        strides=(2, 2)
    ),
    
    # Second conv layer.
    # This layer has 16 filters, each 5x5.
    # 16 filters to learn richer patterns (combinations of strokes).
    tf.keras.layers.Conv2D(
        kernel_size=5,
        filters=16,
        strides=1,
        padding='same',
        activation=tf.keras.activations.relu, # non linearity function
        kernel_initializer=tf.keras.initializers.VarianceScaling()    
    ),
    tf.keras.layers.MaxPool2D(
        pool_size=(2, 2),
        strides=(2, 2)
    ),
    
    # Flatten the 3D output to 1D for the dense layers.
    # Flatten turns feature maps into a vector.
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation="softmax"),
])

# print a summary of the model’s layers.
model.summary()

# compile the model
model.compile(
    # The optimizer is how the model updates its weights after each batch.
    # 0.001 is the standard default learning rate.
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # Adam = a version of gradient descent
    loss=tf.keras.losses.sparse_categorical_crossentropy, # loss function for multi-class classification
    metrics=['accuracy'] # report accuracy during training
)

log_dir = ".logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

training_history = model.fit(
    x_train_with_channels,
    y_train,
    epochs=10,
    validation_data=(x_test_with_channels, y_test),
    callbacks=[tensorboard_callback]
)

# Evaluate model accuracy 
# Training accuracy 
train_loss, train_accuracy = model.evaluate(x_train_with_channels, y_train, verbose=2)
print('\nTraining accuracy:', train_accuracy)
print('Training loss:', train_loss)

# Test accuracy
test_loss, test_accuracy = model.evaluate(x_test_with_channels, y_test, verbose=2)
print('\nTest accuracy:', test_accuracy)
print('Test loss:', test_loss)

# save the model
model.save('saved_models/mnist_cnn_model.keras')