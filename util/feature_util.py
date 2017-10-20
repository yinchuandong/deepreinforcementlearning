from PIL import Image
import numpy as np


def create_process_fn(is_atari=True, use_rgb=False):
    """ preprocess inputted image according to different games
    Args:
        is_atari: boolean, whether it's atari game
        use_rgb: boolean, whether use rgb or gray image
    Returns:
        f: function
    """
    if is_atari:
        scale_size = (84, 110)
        crop_area = (0, 20, 84, 104)
    else:
        scale_size = (84, 110)
        crop_area = (0, 0, 84, 84)
        # crop_area = None

    def f(img_array):
        img = Image.fromarray(img_array)
        # img = img.resize(scale_size, Image.ANTIALIAS)  # blurred
        img = img.resize(scale_size)
        if crop_area is not None:
            img = img.crop(crop_area)
        if not use_rgb:
            img = img.convert('L')
            return np.reshape(img, (img.size[1], img.size[0], 1))
        else:
            return np.array(img)
    return f


def normalize(img):
    """ normalize a image to [-1, 1]
    Args:
        img: (np.array), shape=[x,x,x]
    Returns:
        np.array
    """
    img = img / 255.0
    # img_mean = np.mean(img)
    # img = img - img_mean
    return img
