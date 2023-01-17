import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
from tensorflow import keras
from keras import layers
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

from config import conf
from extract import extract_test

true_model = keras.models.load_model(conf["modelPath"])


model = keras.Sequential(
    [
        keras.Input(shape=conf["inputShape"]),
        true_model.layers[0],
        # true_model.layers[1],
        # true_model.layers[2],
        # true_model.layers[3],
        # true_model.layers[4],
        # true_model.layers[5],
        # true_model.layers[6],
    ]
)


X, y = extract_test()
prediction = model.predict(np.expand_dims(X[0], 0))
print(prediction.shape)
imgs = []
for i in range(prediction.shape[3]):
    slide = prediction[0, :, :, i]
    img = Image.fromarray(slide.astype(np.uint8))
    imgs.append(img)


if prediction.shape[3] == 32:
    ar = np.ones((280, 140)) * 255
else:
    ar = np.ones((280, 280)) * 255

full = Image.fromarray(ar)

i = 0
j = 0
for img in imgs:
    size = int(prediction.shape[2] * 34 / prediction.shape[2])
    img = img.resize((size, size), resample=0)
    full.paste(img, ((i * size) + i, (j * size) + j))
    i += 1
    if (prediction.shape[3] == 32 and i == 4) or (i == 8):
        i = 0
        j += 1

full = full.convert("RGB")
cm = plt.get_cmap('inferno')
test = np.array(full)
test = cm(test)
test = test[:, :, 0, :3] * 255
full = Image.fromarray(test.astype(np.uint8))
# full.show()
full.save("assets/layer1.png")