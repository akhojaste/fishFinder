import tensorflow as tf
from dataset import *
import matplotlib
matplotlib.use('TKAgg', warn=False, force=True)
from matplotlib import pyplot as plt
from config import *

class augment:

    def __init__(self):
        pass

    def __call__(self, image, is_training=False):

        if is_training:
            ## TODO, for random_crop to work, the input image size needs to be bigger than
            ## the crop size, this might fail, which we can increase crop_length to resolve it
            ## but then not too much since it loweres the resolution
            image = tf.image.random_crop(image, [IMAGE_SIZE - CROP_LENGTH, IMAGE_SIZE - CROP_LENGTH, 3])
            image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
            image = tf.image.random_flip_left_right(image)
            image -= MEAN
            image /= STD
            image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
            return image
        else:
            image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
            image -= MEAN
            image /= STD
            return image


if __name__ == "__main__":
    inData = InputData()
    trainDs, testDs = inData.getDatasets()

    # dataset has not make_one_shot_iterator anymore in tf 2.0!!!
    itr = tf.compat.v1.data.make_one_shot_iterator(trainDs)
    image, label = itr.get_next()
    # Resshape
    image = tf.squeeze(image[0, :, :, :])
    plt.imshow(image)
    plt.show()