# MNIST
A recreation of the MNIST convolutional neural network.

## Background

---

## Usage

```
usage: MNIST [-h] {display,predict,retrain} ...

Implementation of the MNIST CNN

positional arguments:
  {display,predict,retrain}
                        command to execute
    display             Display the MNIST image with the given id
    predict             Predict the value of the MNIST image with the given id
    retrain             Retrain the model

options:
  -h, --help            show this help message and exit



usage: MNIST display [-h] [--train] [id]

positional arguments:
  id          ID of the MNIST image to display, randomized if not provided

options:
  -h, --help  show this help message and exit
  --train     Flag to specify using the train dataset instead of the test dataset



usage: MNIST predict [-h] [--train] [-d] [id]

positional arguments:
  id             ID of the MNIST image to predict, randomized if not provided

options:
  -h, --help     show this help message and exit
  --train        Flag to specify using the train dataset instead of the test dataset
  -d, --display  Flag to display the image with prediction



usage: MNIST retrain [-h] [-s]

options:
  -h, --help  show this help message and exit
  -s, --save  Flag to save the retrained model
```

---

## Project Sutructure

```
├── MNIST
│   ├── data
│   │   ├── mnist_test.csv
│   │   ├── mnist_train1.csv
│   │   └── mnist_train2.csv
│   ├── model
│   ├── src
│   │   ├── config.py
│   │   ├── extract.py
│   │   ├── main.py
│   │   ├── model.py
│   │   ├── preProcess.py
│   │   └── reTrain.py
│   └── README.md
```

| File | Description |
| --- | --- |
| data/mnist_test.csv | csv containing the test MNIST dataset. column 1 contains the classification. columns 1-785 contain 8bit grayscale image pixel values |
| data/mnist_train1.csv | csv containing the first half of train MNIST dataset. column 1 contains the classification. columns 1-785 contain 8bit grayscale image pixel values |
| data/mnist_train2.csv | csv containing the second half of train MNIST dataset. column 1 contains the classification. columns 1-785 contain 8bit grayscale image pixel values |
| model/ | files storing the model, produced by Keras |
| src/config.py | configuration file |
| src/extract.py | Function to extract data from MNIST csv files |
| src/main.py | main cli function |
| src/model.py | definition for untrained model |
| src/preProcess.py | image data preprocessing function |
| src/reTrain.py | function to retrain the model and display training data |
| README.md | this README file |
