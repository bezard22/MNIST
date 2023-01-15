# src/preProcess.py

import numpy as np
from tensorflow import keras

from config import conf

def preProcess(X, y):

    # Scale images to the [0, 1] range
    X = X.astype("float32") / 255
    
    # Make sure images have shape (28, 28, 1)
    X = np.expand_dims(X, -1)

    # convert class vectors to binary class matrices
    y = keras.utils.to_categorical(y, conf["numClasses"])

    return (X, y)