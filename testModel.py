# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 22:56:00 2019

@author: amir
"""
import os
import tensorflow as tf
import numpy as np
from dataset import InputData
import matplotlib.pyplot as plt
from tensorflow import keras
import imageio

## Input shapes
inputData = InputData()
getLabelName = inputData.getLabelName

def getModel(checkpoint_path):
    """
    When loading the weight into a model, first we have to construct
    that model, hence stacking the layers together and compile
    """
    inputShape = inputData.getImageShape()
    baseModel = tf.keras.applications.MobileNetV2(input_shape=inputShape,
                                                  include_top=False,
                                                  weights='imagenet')
    globalAvgPooling = tf.keras.layers.GlobalAveragePooling2D()
    layerList = [baseModel,
                 globalAvgPooling,
                 tf.keras.layers.Dense(3, activation='softmax')
                 ]
    model = tf.keras.Sequential(layerList)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())

    # Load the weights
    model.load_weights(checkpoint_path)

    return model


def main():

    model = getModel('D:/Fish/checkpoints/ckp_16')

    print('*' * 20, 'Saving model')
    model.save('D:/Fish/checkpoints/model.h5')
    print('*' * 20, 'Saving the model is done.')

    ## ------------ TestDS
    # Evaluate now using the checkpoints
    # loss, acc = model.evaluate(testDs)
    # print('loss {}, acc {}'.format(loss, acc))
    #
    # # test on one batch only
    # for image, label in testDs.take(1):
    #     pass
    #
    # for i in range(0, 32):
    #     img_ = image[i, :, :, :]
    #     img_ = np.expand_dims(img_, 0)
    #     classIdx = model.predict_classes(img_)
    #     print('target : {}, result: {}'.format(np.argmax(label[i, :]), classIdx))

    ## ----------- Custom Image
    file_names = ['D:\Fish\pike\images - 2019-08-15T224501.686.jpg',
                 'D:\Fish\largemouth_bass\9k_.jpg',
                 'D:\Fish\carp\images - 2019-08-15T225156.575.jpg',
                 'D:\Fish\largemouth_bass\images - 2019-08-15T223949.982.jpg']

    images = np.zeros([4, 160, 160, 3])
    for index, file_name in enumerate(file_names):
        # have to do the pre-process on these images
        images[index] = np.expand_dims(inputData.preprocessSingleImage(file_name, False), 0)

    classIdx = model.predict_classes(images)
    for i in range(len(file_names)):
        print(getLabelName(np.squeeze(classIdx[i])))


def export_model():

    model = getModel('D:/Fish/checkpoints/ckp_16')

    print('*' * 20, 'Saving model')
    saved_dir = 'D:/Fish/checkpoints/model.h5'
    model.save(saved_dir)
    print('*' * 20, 'Saving the model is done.')

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open('converted_model.tflite', 'wb').write(tflite_model)



if __name__ == "__main__":
    # main()
    export_model()