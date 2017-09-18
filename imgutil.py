from PIL import Image


def scale_image(img_array, size=(84, 84), gray=False):
    img = Image.fromarray(img_array)
    img = img.resize(size, Image.ANTIALIAS)
    if gray:
        img = img.convert('L')
    return img
