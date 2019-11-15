from PIL import Image


def __init__(self):
    self.scale_img()


def scale_img(input_img, new_size):
    return input_img.resize(new_size, resample=Image.NEAREST)






