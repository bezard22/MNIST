import numpy as np
from PIL import Image, ImageDraw

from extract import extract_test

X, y = extract_test()
ar0 = X[0].astype(np.uint8)
ar1 = np.zeros((532, 532), dtype=np.uint8)

for i in range(28):
    for j in range(28):
        ar1[i*19:(i+1)*19, j*19:(j+1)*19] = np.ones((19, 19)) * ar0[i, j]
        # ar1[i*19:(i+1)*19, j*19:(j+1)*19] = np.ones((19, 19)) * 0

img = Image.fromarray(ar1)
# draw = ImageDraw.Draw(img)
# for i in range(28):
#     for j in range(28):
#         val = 0 if ar0[i, j] > 127 else 255
#         draw.text((1 + j*19, 4 + i*19), str(ar0[i, j]), fill=val)
img.show()
img.save("tmp0.png")