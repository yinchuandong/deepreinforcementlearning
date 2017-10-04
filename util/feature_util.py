from PIL import Image
import numpy as np


def process_image(img_array, scale_size=(84, 110), crop_area=None, use_rgb=False):
    """ scale and crop image

    Args:
        img_array: (numpy.array) the raw image to be processed
        scale_size: (tuple) a tuple of (width, height)
        crop_area: (tuple) a tuple of (left, top, right, bottom)
        use_rgb: (boolean) whether use rgb or gray image
    Returns:
        img: a numpy array, with shape [x, x, 1] for gray and [x, x, 3] for rgb
    """
    img = Image.fromarray(img_array)
    img = img.resize(scale_size, Image.ANTIALIAS)
    if crop_area is not None:
        print(crop_area)
        img = img.crop(crop_area)
    if not use_rgb:
        img = img.convert('L')
        return np.reshape(img, (img.size[1], img.size[0], 1))
    else:
        return np.array(img)
