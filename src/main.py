# src/main.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
from tensorflow import keras
from preProcess import preProcess

from extract import extract_test
from reTrain import reTrain

import pandas as pd


if __name__ == "__main__":
    X, y = extract_test()
    X, y = preProcess(X, y)

    model = keras.models.load_model("model")
    score = model.evaluate(X, y, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
