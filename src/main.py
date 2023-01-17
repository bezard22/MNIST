# src/main.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tensorflow import keras
import numpy as np
from PIL import Image
from random import randint
import argparse

from config import conf
from preProcess import preProcess
from extract import extract_test, extract_train
from reTrain import reTrain

# ------------------------------------------------------------------------
#     main  -  cli for MNIST model
# ------------------------------------------------------------------------

def parse() -> dict[str, any]:
    """parse cmdline arguments

    :return: a dictionary of cmdline arguments and their values
    :rtype: dict[str, any]
    """    
    parser = argparse.ArgumentParser(prog="MNIST", description="Implementation of the MNIST CNN")
    subparsers = parser.add_subparsers(help="command to execute", dest="cmd")

    # display
    displayParser = subparsers.add_parser("display", help="Display the MNIST image with the given id")
    displayParser.add_argument("id",
        help="ID of the MNIST image to display, randomized if not provided",
        type=int,
        default=randint(0, 9999),
        nargs="?"
    )
    displayParser.add_argument("--train",
        help="Flag to specify using the train dataset instead of the test dataset",
        action="store_true"
    )
    displayParser.add_argument("-s", "--save",
        help="Flag to save the image file in the current directory",
        action="store_true"
    )
    
    # predict
    predictParser = subparsers.add_parser("predict", help="Predict the value of the MNIST image with the given id")
    predictParser.add_argument("id",
        help="ID of the MNIST image to predict, randomized if not provided",
        type=int,
        default=randint(0, 9999),
        nargs="?"
    )
    predictParser.add_argument("--train",
        help="Flag to specify using the train dataset instead of the test dataset",
        action="store_true"
    )
    predictParser.add_argument("-d", "--display",
        help="Flag to display the image with prediction",
        action="store_true"
    )

    # retrain
    retrainParser = subparsers.add_parser("retrain", help="Retrain the model")
    retrainParser.add_argument("-s", "--save",
        help="Flag to save the retrained model",
        action="store_true"
    )

    return vars(parser.parse_args())

def display(n: int, train=False, save=False) -> None:
    """display the MNIST image based on id and split

    :param n: id (row) of image to display
    :type n: int
    :param train: flag to use training set instead of testing set, defaults to False
    :type train: bool, optional
    :param save: , defaults to False
    :type save: bool, optional
    """    
    split = "train" if train else "test"
    print(f"Displaying id: {n} from {split}")
    if split == "test":
        X, y = extract_test()
    else:
        X, y = extract_train()
    img = Image.fromarray(X[n].astype(np.uint8))
    img.show()
    
    if save:
        img.save(f"{n}.png")
        print(f"saved: {n}.png")

def predict(n: str, train=False) -> None:
    """use the model to analyse an image and predict the digit

    :param n: id (row) of image to display
    :type n: str
    :param train: flag to use training set instead of testing set, defaults to False
    :type train: bool, optional
    """    
    split = "train" if train else "test"
    print(f"Predicting id: {n} from {split}")
    if split == "test":
        X, y = extract_test()
    else:
        X, y = extract_train()
    print(f"classification: {y[n]}")
    X, y = preProcess(X, y)
    model = keras.models.load_model(conf["modelPath"])
    prediction = model.predict(np.expand_dims(X[n], 0))
    print(f"prediction: {np.argmax(prediction[0])}")


if __name__ == "__main__":
    args = parse()

    if args["cmd"] == "display":
        display(args["id"],args["train"], args["save"])
    elif args["cmd"] == "predict":
        if args["display"]:
            display(args["id"],args["train"])
        predict(args["id"], train=args["train"])
    elif args["cmd"] == "retrain":
        reTrain(args["save"])