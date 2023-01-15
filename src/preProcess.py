# src/preProcess.py

import numpy
from tensorflow import keras

def preProcess(X_train, y_train, X_test, y_test):
    num_classes = 10

    # Scale images to the [0, 1] range
    X_train = X_train.astype("float32") / 255
    X_test = X_test.astype("float32") / 255
    
    # Make sure images have shape (28, 28, 1)
    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return (X_train, y_train, X_test, y_test)