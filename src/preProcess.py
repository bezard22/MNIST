# src/preProcess.py

import numpy as np
from tensorflow import keras

from config import conf

# ------------------------------------------------------------------------
#     preProcess  -  input preProcessing function
# ------------------------------------------------------------------------

def preProcess(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """preprocess image datasets

    :param X: X dataset
    :type X: np.ndarray
    :param y: y dataset
    :type y: np.ndarray
    :return: tuple of preprocessed X and y datasets
    :rtype: tuple[np.ndarray, np.ndarray]
    """    
    
    # Scale images to the [0, 1] range
    X = X.astype("float32") / 255
    
    # Make sure images have shape (28, 28, 1)
    X = np.expand_dims(X, -1)

    # convert class vectors to binary class matrices
    y = keras.utils.to_categorical(y, conf["numClasses"])

    return (X, y)