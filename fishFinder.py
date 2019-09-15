# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 22:56:00 2019

@author: amir
"""
from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
import dataset
import matplotlib.pyplot as plt
from tensorflow import keras
import imageio
from functools import partial


def main():
    
    ## Input
    inputData = dataset.InputData()
    # iterator, trainInitOp, testInitOp = inputData.getIterator()
    # image, label = iterator.get_next()

    trainDs, testDs = inputData.getDatasets()

    inputShape = inputData.getImageShape()
    baseModel = tf.keras.applications.MobileNetV2(input_shape=inputShape,
                                                  include_top=False,
                                                  weights='imagenet')

    print(baseModel.summary())

    for image, label in trainDs.take(1):
        print('image shape {}, label shape {}'.format(image.shape, label.shape))

    featureMap = baseModel(image)
    print('base model feature shape: {}'.format(featureMap.shape))

    #Freeze the base, one way is to set the trainable
    # variable on the whole model
    # baseModel.trainable = False
    print('number of base model layers {}'.format(len(baseModel.layers)))
    baseModel.trainable = True
    fineTuneAt = len(baseModel.layers) - 3
    for layer in baseModel.layers[:fineTuneAt]:
        layer.trainable = False

    # for i, layer in enumerate(baseModel.layers):
    #
    #     if i < (len(baseModel.layers) - 3):
    #         layer.trainable = False
    #     else:
    #         # training one last conv layer
    #         layer.trainable = True

    globalAvgPooling = tf.keras.layers.GlobalAveragePooling2D()
    avgPooled = globalAvgPooling(featureMap)
    print('avgPooled shape : {}'.format(avgPooled.shape))

    layerList = [baseModel,
                 globalAvgPooling,
                 # tf.keras.layers.Dropout(0.5),
                 tf.keras.layers.Dense(128, activation='relu'),
                 tf.keras.layers.Dropout(0.5),
                 # tf.keras.layers.Dense(64, activation='relu'),
                 #Since this is multi-class classification, last layer should have softmax activation
                 #in case of binary classification, we can ignore this or set sigmoid
                 tf.keras.layers.Dense(dataset.NUM_CLASSES,
                                       kernel_regularizer=keras.regularizers.l2(30.0),
                                       activation=keras.activations.softmax)
                 ]

    model = tf.keras.Sequential(layerList)

    # To apply label smoothing, need to make the partial function
    loss_label_smoothed = partial(keras.losses.categorical_crossentropy, label_smoothing=0.2)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  loss=keras.losses.categorical_crossentropy,
                  # loss=loss_label_smoothed,
                  metrics=['accuracy'])

    print(model.summary())


    def scheduler(epoch):
        if epoch < 3:
            return 0.01
        else:
            return 0.005
        # elif epoch >= 3 and epoch < 10:
        #     return 0.001
        # elif epoch >= 10:
        #     return 0.0001

    # def scheduler(epoch):
    #     init_lr = 0.001
    #     lr = 0.5 * (1 + tf.cos(epoch * 3.1415 / 50)) * init_lr
    #     return float(lr)

    # Training
    checkpointPath = os.path.join('D:/Fish/checkpoints', 'ckp_{epoch}')
    callbacks = [keras.callbacks.LearningRateScheduler(scheduler),
                 keras.callbacks.TensorBoard(),
                 keras.callbacks.ModelCheckpoint(checkpointPath, save_weights_only=True)]

    # Restore from saved epoch
    # init_epoch = 16
    # model.load_weights('D:/Fish/checkpoints/ckp_{}'.format(init_epoch))
    # Check and see if the model is loaded correctly
    # loss, acc = model.evaluate(testDs)
    # print('loss {} and acc {}'.format(loss, acc))

    # model.fit(trainDs,
    #           validation_data=testDs,
    #           epochs=50,
    #           callbacks=callbacks,
    #           initial_epoch=init_epoch)

    model.fit(trainDs,
              validation_data=testDs,
              epochs=40,
              callbacks=callbacks)

    # To save the entire model
    # model.save('D:/Fish/checkpoints/model.h5')
    # Evaluation, sine both test and validations are the same
    # we skip this area
    # loss, acc = model.evaluate(testDs)
    # print('loss {}, acc {}'.format(loss, acc))


if __name__ == "__main__":
    main()
