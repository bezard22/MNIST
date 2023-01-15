# src/extract.py

import numpy as np
import pandas as pd

from config import conf

def extract_test():
    test = pd.read_csv(conf["testPath"], engine="c", header=None).to_numpy()
    X_test = test[:, 1:]
    y_test = test[:, 1]
    X_test = np.reshape(X_test, (-1, 28, 28))
    return (X_test, y_test)

def extract_train():
    train1 = pd.read_csv(conf["train1Path"], engine="c", header=None)
    train2 = pd.read_csv(conf["train2Path"], engine="c", header=None)
    train = pd.concat([train1, train2]).to_numpy()
    X_train = train[:, 1:]
    y_train = train[:, 1]
    X_train = np.reshape(X_train, (-1, 28, 28))
    return (X_train, y_train)