from tensorflow import keras

# load the saved model
model = keras.models.load_model('saved_models/mnist_cnn_model.keras')