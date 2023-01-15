# src/extract.py

import numpy as np
import pandas as pd

from config import conf

# ------------------------------------------------------------------------
#     extract  -  extract image datasets from csv files
# ------------------------------------------------------------------------

def extract_test() -> tuple[np.ndarray, np.ndarray]:
    """extract test dataset.

    :return: tuple of X and y datasets
    :rtype: tuple[np.ndarray, np.ndarray]
    """    
    test = pd.read_csv(conf["testPath"], engine="c", header=None).to_numpy()
    X_test = test[:, 1:]
    y_test = test[:, 0]
    X_test = np.reshape(X_test, (-1, 28, 28))
    return (X_test, y_test)

def extract_train() -> tuple[np.ndarray, np.ndarray]:
    """extract train dataset.

    :return: tuple of X and y datasets
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    train1 = pd.read_csv(conf["train1Path"], engine="c", header=None)
    train2 = pd.read_csv(conf["train2Path"], engine="c", header=None)
    train = pd.concat([train1, train2]).to_numpy()
    X_train = train[:, 1:]
    y_train = train[:, 0]
    X_train = np.reshape(X_train, (-1, 28, 28))
    return (X_train, y_train)