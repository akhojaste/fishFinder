# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 22:56:00 2019

@author: amir
"""
import os
import tensorflow as tf
import numpy as np
from tensorflow_core.python.tools.api.generator.create_python_api import _get_name_and_module
from tensorflow_estimator.python.estimator.canned.dnn import _name_from_scope_name

from dataset import InputData
import matplotlib.pyplot as plt
import imageio
from config import *
import fishFinder

CHECKPOINT_PATH = './checkpoints/ckp_100    '

def load_model(input_shape):
    """
    When loading the weight into a model, first we have to construct
    that model, hence stacking the layers together and compile
    :param input_shape: shape of the input to the network
    :return loaded model.
    """
    model = fishFinder.get_model(input_shape)
    # Load the weights
    model.load_weights(CHECKPOINT_PATH)
    return model


def inference():
    """
    Runs an inference on the test dataset
    :param: None
    :return: None
    """
    # Input shapes
    input_data = InputData()
    input_shape = input_data.getImageShape()
    model = load_model(input_shape)

    # get the test data
    train_gen, valid_gen, test_gen = input_data.get_keras_ds()
    # getting the label map, NOTE, we have to get the label_map from
    # generator since we used flow_from_directory and did not specify
    # the label_map, if we use test_gen however, since it is using
    # flow_from_directory and the folder names are the same in test folder
    # then we don't even need this label_map
    label_map = train_gen.class_indices
    label_map = dict((v, k) for k, v in label_map.items())  # flip k,v

    accs = []
    for step, (image, label) in enumerate(test_gen):
        print('image.shape: {}, label.shape{}'.format(image.shape, label.shape))
        class_idx_batch = model.predict_classes(image)
        acc = np.sum(np.equal(np.argmax(label, 1), class_idx_batch)) / label.shape[0]
        print('batch acc: {}'.format(acc))
        # we need to break out of loop since generator loops infinitely
        if image.shape[0] < BATCH_SIZE:
            # calculate the final accuracy over all the test data
            tmp = np.sum(accs) * BATCH_SIZE + acc * image.shape[0]
            total_acc = tmp / (step * BATCH_SIZE + image.shape[0])
            print('Total acc on whole test data: {}'.format(total_acc))
            break

        accs.append(acc)


    # Custom images
    # file_names = ['./DATA/test/crappie/images (53).jpg',
    #               './DATA/test/largemouth_bass/2Q__ (3).jpg',
    #               './DATA/test/pike/2Q__ (4).jpg',
    #               './DATA/test/carp/2Q__ (1).jpg']
    #
    # images = np.zeros([4, IMAGE_SIZE, IMAGE_SIZE, 3])
    # for index, file_name in enumerate(file_names):
    #     # have to do the pre-process on these images
    #     images[index] = np.expand_dims(input_data.preprocessSingleImage(file_name, False), 0)
    #
    # class_idx = np.squeeze(model.predict_classes(images))
    # for i in range(len(file_names)):
    #     print(label_map[class_idx[i]])


def export_model():
    """
    Exports the TFLite format of the saved model
    :param: None
    :return: None
    """
    # Input shapes
    input_data = InputData()
    input_shape = input_data.getImageShape()

    model = load_model(input_shape)

    print('*' * 20, 'Saving model')
    saved_dir = './Deploy/TFLite/new_model.h5'
    model.save(saved_dir)
    print('*' * 20, 'Saving the model is done.')

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open('./Deploy/TFLite/converted_model.tflite', 'wb').write(tflite_model)


if __name__ == "__main__":
    # inference on some test images
    inference()

    # export model to use on cellphone
    export_model()
