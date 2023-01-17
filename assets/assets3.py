import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
from tensorflow import keras
from keras import layers
from PIL import Image, ImageDraw
import pandas as pd
import matplotlib.pyplot as plt

from config import conf
from extract import extract_test

true_model = keras.models.load_model(conf["modelPath"])


model = keras.Sequential(
    [
        keras.Input(shape=conf["inputShape"]),
        true_model.layers[0],
        true_model.layers[1],
        true_model.layers[2],
        true_model.layers[3],
        true_model.layers[4],
        true_model.layers[5],
        # true_model.layers[6],
    ]
)


X, y = extract_test()
prediction = model.predict(np.expand_dims(X[0], 0))
print(prediction.shape)
prediction = np.reshape(prediction, (1, 40, 40))
print(prediction.shape)

slide = prediction[0, :, :]
img = Image.fromarray(slide.astype(np.uint8))
full = img.resize((280, 280), resample=0)

full = full.convert("RGB")
cm = plt.get_cmap('inferno')
test = np.array(full)
test = cm(test)
test = test[:, :, 0, :3] * 255
full = Image.fromarray(test.astype(np.uint8))
# full.show()
full.save("assets/layer6.png")


# prediction = np.transpose(prediction)
# slide = prediction[:, 0] * 255
# img = Image.fromarray(slide.astype(np.uint8))
# full = img.resize((140, 280), resample=0)

# draw = ImageDraw.Draw(full)
# for i in range(10):
#     draw.line([5, (i + 1) * 28, full.size[0], (i + 1) * 28], fill=127)
#     draw.text((5, i * 28 + 8), str(i), fill=127)

# full = full.convert("RGB")
# cm = plt.get_cmap('inferno')
# test = np.array(full)
# test = cm(test)
# test = test[:, :, 0, :3] * 255
# full = Image.fromarray(test.astype(np.uint8))
# # full.show()
# full.save("assets/layer7.png")