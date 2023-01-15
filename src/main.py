# src/main.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tensorflow import keras
import numpy as np
from PIL import Image
from random import randint

from config import conf
from preProcess import preProcess
from extract import extract_test
from reTrain import reTrain


def display(ar):
    img = Image.fromarray(ar.astype(np.uint8))
    img.show()

def predict(ar):
    model = keras.models.load_model(conf["modelPath"])
    prediction = model.predict(np.expand_dims(ar, 0))
    print(np.argmax(prediction[0]))


if __name__ == "__main__":
    n = randint(0, 9000)
    X, y = extract_test()
    print(y[n])
    display(X[n])
    X, y = preProcess(X, y)
    predict(X[n])

    # model = keras.models.load_model("model")
    # score = model.evaluate(X, y, verbose=0)
    # print("Test loss:", score[0])
    # print("Test accuracy:", score[1])
