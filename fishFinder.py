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
# import matplotlib.pyplot as plt
from tensorflow import keras
# import imageio
from functools import partial
from config import *
import PIL
from PIL import Image


def get_base(input_shape):
    """
    Returns the base model which is pre-trained on imagenet
    :param input_shape: shape of input to the network
    :return: pre-trained model
    """
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                   include_top=False,
                                                   weights='imagenet')
    return base_model


def get_model(input_shape):
    """
    constructs the model and returns it
    :param: input_shape: shape of the input to the network
    :return: model
    """
    # Base model
    base_model = get_base(input_shape)
    # print(base_model.summary())

    # feature_map = base_model(image)
    # print('base model feature shape: {}'.format(feature_map.shape))

    # Freeze the base, one way is to set the trainable
    # variable on the whole model
    # base_model.trainable = False
    print('number of base model layers {}'.format(len(base_model.layers)))
    base_model.trainable = True
    fine_tune_at = len(base_model.layers) - LAYERS_TO_TRAIN
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    global_avg_pooling = tf.keras.layers.GlobalAveragePooling2D()
    # avg_pooled = global_avg_pooling(feature_map)
    # print('avg_pooled shape : {}'.format(avg_pooled.shape))

    layer_list = [base_model,
                  global_avg_pooling,
                  tf.keras.layers.Dense(128, activation='relu'),
                  tf.keras.layers.Dropout(0.1),
                  # tf.keras.layers.Dense(64, activation='relu'),
                  # Since this is multi-class classification, last layer should have softmax activation
                  # in case of binary classification, we can ignore this or set sigmoid
                  tf.keras.layers.Dense(dataset.NUM_CLASSES,
                                        kernel_regularizer=keras.regularizers.l2(L2_REG),
                                        activation=keras.activations.softmax)
                  ]

    model = tf.keras.Sequential(layer_list)

    # To apply label smoothing, need to make the partial function
    loss_label_smoothed = partial(keras.losses.categorical_crossentropy, label_smoothing=0.2)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  loss=keras.losses.categorical_crossentropy,
                  # loss=loss_label_smoothed,
                  metrics=['accuracy'])

    return model


def main():
    
    # Input
    input_data = dataset.InputData()
    # train_ds, test_ds = input_data.getDatasets()
    # for image, label in train_ds.take(1):
    #     print('image shape {}, label shape {}'.format(image.shape, label.shape))

    # Model
    model = get_model(input_data.getImageShape())
    # print(model.summary())

    def scheduler(epoch):
        if epoch < 5:
            return 0.01
        elif epoch >= 5 and epoch < 10:
            return 0.001
        elif epoch >= 10:
            return 0.0001

    # cosine learning rate
    # def scheduler(epoch):
    #     init_lr = 0.01
    #     lr = 0.5 * (1 + tf.cos(epoch * 3.1415 / EPOCH)) * init_lr
    #     return float(lr)

    # Training
    checkpoint_path = os.path.join('checkpoints', 'ckp_{epoch}')
    callbacks = [keras.callbacks.LearningRateScheduler(scheduler),
                 keras.callbacks.TensorBoard(),
                 keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True),
                 # keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
                 keras.callbacks.ReduceLROnPlateau(patience=3)
                ]

    # Restore from saved epoch
    # init_epoch = 16
    # model.load_weights('D:/Fish/checkpoints/ckp_{}'.format(init_epoch))
    # Check and see if the model is loaded correctly
    # loss, acc = model.evaluate(test_ds)
    # print('loss {} and acc {}'.format(loss, acc))

    # model.fit(train_ds,
    #           validation_data=test_ds,
    #           epochs=50,
    #           callbacks=callbacks,
    #           initial_epoch=init_epoch)

    # model.fit(train_ds,
    #           validation_data=test_ds,
    #           epochs=EPOCH,
    #           callbacks=callbacks)

    # Using keras generator
    train_generator, val_generator, test_generator = input_data.get_keras_ds()
    model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator.filenames) // BATCH_SIZE,
        epochs=EPOCH,
        validation_data=val_generator,
        validation_steps=len(train_generator.filenames) // BATCH_SIZE,
        # class_weight=class_weights,
        callbacks=callbacks
    )

    # To save the entire model
    # model.save('D:/Fish/checkpoints/model.h5')
    # Evaluation, sine both test and validations are the same
    # we skip this area
    # loss, acc = model.evaluate(test_ds)
    # print('loss {}, acc {}'.format(loss, acc))


if __name__ == "__main__":
    main()
