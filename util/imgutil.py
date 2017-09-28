from PIL import Image
import numpy as np


def scale_image(img_array, size=(84, 84), use_rgb=False):
    img = Image.fromarray(img_array)
    img = img.resize(size, Image.ANTIALIAS)
    if not use_rgb:
        img = img.convert('L')
        return np.reshape(img, (size[0], size[1], 1))
    else:
        return np.array(img)
