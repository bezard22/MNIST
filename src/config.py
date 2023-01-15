# src/config.py

# ------------------------------------------------------------------------
#     config  -  global config values
# ------------------------------------------------------------------------

conf = {
    "testPath": "data/mnist_test.csv",      # path to test csv
    "train1Path": "data/mnist_train1.csv",  # path to train1 csv
    "train2Path": "data/mnist_train2.csv",  # path to train2 csv
    "modelPath": "model",                    # path to saved model
    "numClasses": 10,                       # number of classes (0-9)
    "inputShape": (28, 28, 1),              # input shape, (28, 28, 1)
    "batchSize": 128,                       # batch size for training
    "epochs": 15                           # # of epochs to perform
}