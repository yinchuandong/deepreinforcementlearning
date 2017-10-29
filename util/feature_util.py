from PIL import Image
import numpy as np


def create_process_fn(env_mode="atari", use_rgb=False):
    """ preprocess inputted image according to different games
    Args:
        env_mode: string, ["atari", "ple", "custom"]
        use_rgb: boolean, whether use rgb or gray image
    Returns:
        f: function
    """
    if env_mode == "atari":
        scale_size = (84, 110)
        crop_area = (0, 20, 84, 104)
    elif env_mode == "ple" or env_mode == "custom":
        scale_size = (84, 110)
        crop_area = (0, 0, 84, 84)
    else:
        raise ValueError("wrong value of env_mode")

    def f(img_array):
        img = Image.fromarray(img_array)
        # img = img.resize(scale_size, Image.ANTIALIAS)  # blurred
        img = img.resize(scale_size)
        if crop_area is not None:
            img = img.crop(crop_area)
        if not use_rgb:
            # img = img.convert('L')
            img = img.convert('L').point(lambda p: p > 100 and 255)
            # img = img.convert('L').point(lambda p: p > 100)
            img = np.reshape(img, (img.size[1], img.size[0], 1))
            return img.astype(np.uint8)
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
