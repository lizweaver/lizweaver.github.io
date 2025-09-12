import numpy as np
import skimage.io as io

def load_image(image_path):
    image = io.imread(image_path)

    if image.dtype == np.uint8:
        image = image.astype(np.float64) / 255.0
    elif image.dtype == np.uint16:
        image = image.astype(np.float64) / 65535.0
    else:
        image = image.astype(np.float64)
        if image.max() > 1.0:
            image = image / image.max()
    
    return image

def split_channels(image):
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)
    
    height = image.shape[0]
    channel_height = height // 3

    blue = image[:channel_height, :]
    green = image[channel_height:2*channel_height, :]
    red = image[2*channel_height:3*channel_height, :]
    
    return blue, green, red

def crop_image(image, border_crop_fraction):
    height, width = image.shape
    crop_height = int(height * border_crop_fraction)
    crop_width = int(width * border_crop_fraction)
    
    return image[crop_height:height-crop_height, crop_width:width-crop_width]
