import tensorflow as tf
from dataset import *
# import matplotlib
# matplotlib.use('TKAgg', warn=False, force=True)
# from matplotlib import pyplot as plt
from config import *
# from autoaugment import ImageNetPolicy
# from PIL import Image
import numpy as np

class augment:

    def __init__(self):
        pass

    def __call__(self, image, is_training=False):

        if is_training:
            # TODO, for random_crop to work, the input image size needs to be bigger than
            # the crop size, this might fail, which we can increase crop_length to resolve it
            # but then not too much since it loweres the resolution
            image = tf.image.random_crop(image, [IMAGE_SIZE - CROP_LENGTH, IMAGE_SIZE - CROP_LENGTH, 3])
            image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
            image = tf.image.random_flip_left_right(image)

            # image = self.tf_augment(image)
            # image = tf.convert_to_tensor(image)

            image -= MEAN
            image /= STD
            image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
            return image
        else:
            image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
            image -= MEAN
            image /= STD
            return image

    @tf.function # what is tf.numpy_function
    def tf_augment(self, image):
        image = tf.py_function (self.policy_augment, inp=[image], Tout=[tf.float32])
        return image

    def policy_augment(self, image):
        image_np = np.asarray(image)
        policy = ImageNetPolicy()
        image_pil = self.pil_wrap(image_np)
        image_pil = policy(image_pil)
        image = self.pil_unwrap(image_pil)
        return image

    def pil_wrap(self, img):
        """Convert the `img` numpy tensor to a PIL Image."""
        return Image.fromarray(np.uint8(img)).convert('RGBA')

    def pil_unwrap(self, pil_img):
        """
        Converts the PIL img to a numpy array.
        """
        pic_array = (np.array(pil_img.getdata()).reshape((160, 160, 3)) / 255.0)
        i1, i2 = np.where(pic_array[:, :, 3] == 0)
        pic_array = (pic_array[:, :, :3] - MEANS) / STDS
        pic_array[i1, i2] = [0, 0, 0]
        return pic_array

if __name__ == "__main__":
    input_data = InputData()
    train_ds, test_ds = input_data.getDatasets()

    for image, label in train_ds:
        image = tf.squeeze(image[0, :, :, :])
        plt.imshow(image)
        plt.show()

        break
