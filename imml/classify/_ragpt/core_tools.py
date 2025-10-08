# License: BSD-3-Clause

from PIL import Image


def resize_image(img, size=(384, 384)):
    return img.resize(size, Image.BILINEAR)
