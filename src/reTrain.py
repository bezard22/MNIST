# src/reTrain.py

from tensorflow import keras

from config import conf
from preProcess import preProcess
from model import model
from extract import extract_test, extract_train


def reTrain(save=False, kerasData=False):
    if kerasData:
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
        X_train, y_train = preProcess(X_train, y_train)
        X_test, y_test = preProcess(X_test, y_test)
    else:
        X_test, y_test = extract_test()
        X_train, y_train = extract_train()
        X_test, y_test = preProcess(X_test, y_test)
        X_train, y_train = preProcess(X_train, y_train)


    model.fit(X_train, y_train, batch_size=conf["batchSize"], epochs=conf["epochs"], validation_split=0.1)

    score = model.evaluate(X_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    if save:
        model.save("model")