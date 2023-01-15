# src/extract.py

import numpy as np
from PIL import Image as im
from os import mkdir
from os.path import join
from tqdm import tqdm


def extract(srcPath, dstPath):
    ar = np.genfromtxt(srcPath, delimiter=',', dtype=np.uint8)
    mkdir(dstPath)
    for i in range(10):
        mkdir(join(dstPath, str(i)))

    for i in tqdm(range(len(ar))):
        img = im.fromarray(np.reshape(ar[i, 1:], (28, 28)))
        img.save(join(dstPath, str(ar[i, 0]), str(i) + ".png"))
    

if __name__ == "__main__":
    srcPath = r"dataset/mnist_test.csv"
    dstPath = r"dataset/test"
    extract(srcPath, dstPath)