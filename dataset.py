# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 09:49:21 2019

@author: amir
"""
import tensorflow as tf
import os
from os import walk
from augment import *
from config import *
import numpy as np
from PIL import Image
from autoaugment import *

class InputData:
    
    def __init__(self):
        """
        Arguments:
            None
        Returns:
            None
        """
        self.trainTestSplit = TRAIN_TEST_SPLIT
        self.batchSize = BATCH_SIZE
        self._createTrainTestDataset()
        
        
    def _getFileNames(self, root):

        fileNames = []
        labels = []

        for idx , label in enumerate(CLASSES):
            for (dirpath, dirnames, names) in walk(os.path.join(TRAIN_ROOT, label)):
                for file in names:                
                    fileNames.append(os.path.join(dirpath, file))
                    labels.append(idx) #One hot it later

        return fileNames, labels
        

    def _createTrainTestDataset(self):
        
        self.fileNames, self.labels = self._getFileNames(TRAIN_ROOT)

        ## Split the data
        fileTrain, labelTrain, fileTest, labelTest = self._splitTrainTest()
        
        ## Train
        self.trainDataset = tf.data.Dataset.from_tensor_slices((fileTrain, labelTrain))
        self.trainDataset = self.trainDataset.map(self._parse_fn_train)
        self.trainDataset = self.trainDataset.shuffle(buffer_size=10000).batch(self.batchSize)
        
        
        ## Test
        self.testDataset = tf.data.Dataset.from_tensor_slices((fileTest, labelTest))
        self.testDataset = self.testDataset.map(self._parse_fn_test)
        self.testDataset = self.testDataset.shuffle(buffer_size=10000).batch(self.batchSize)

    def _splitTrainTest(self):

        fileTrain, labelTrain, fileTest, labelTest = [], [], [], []
        for n in range(len(CLASSES)):
            
            numSamples = 0
            if n == len(CLASSES) - 1:
                numSamples = len(self.labels) - self.labels.index(n)
            else:
                numSamples = self.labels.index(n + 1) - self.labels.index(n)
                
            #Now put the first 20% in test and the rest in train
            offset = self.labels.index(n)
            fileTest.extend(self.fileNames[offset : offset + int(self.trainTestSplit * numSamples)])
            labelTest.extend(self.labels[offset : offset + int(self.trainTestSplit * numSamples)])
            
            fileTrain.extend(self.fileNames[offset + int(self.trainTestSplit * numSamples) : offset + numSamples])
            labelTrain.extend(self.labels[offset + int(self.trainTestSplit * numSamples) : offset + numSamples])
            
        self._total_train_size = len(fileTrain)
        self._total_test_size = len(fileTest)
        return fileTrain, labelTrain, fileTest, labelTest

    def _parse_fn_train(self, fileName, label):
        """
        Parses the filenames and labels and return image, label tuple
        Arguments:
            fileName = a single file path
            label = a single label, its not one hot encoded yet
        Return:
            image: a decoded image
            label: one-hot encoded label
        """
        imageFinal = self.preprocessSingleImage(fileName, True)
        return imageFinal, tf.one_hot(label, depth=NUM_CLASSES)

    def _parse_fn_test(self, fileName, label):
        """
        Parses the filenames and labels and return image, label tuple
        Arguments:
            fileName = a single file path
            label = a single label, its not one hot encoded yet
        Return:
            image: a decoded image
            label: one-hot encoded label
        """
        imageFinal = self.preprocessSingleImage(fileName, False)
        return imageFinal, tf.one_hot(label, depth=NUM_CLASSES)
        
    def getIterator(self):
        """
        Return the iterator and the train/test init ops
        """
        
        ## Reinitializable iterator
        iterator = tf.data.Iterator.from_structure(self.trainDataset.output_types, 
                                                        self.trainDataset.output_shapes)
        
        trainInitOp = iterator.make_initializer(self.trainDataset)
        testInitOp = iterator.make_initializer(self.testDataset)  
        
        
        return iterator, trainInitOp, testInitOp

    def getDatasets(self):
        """
        Returns the training, validation and test datasets

        """

        return self.trainDataset, self.testDataset

    def getImageShape(self):
        """

        :return: shape of the input image
        """

        return (IMAGE_SIZE, IMAGE_SIZE, 3)

    def getLabelName(self, idx):

        assert (idx >=0 and idx < len(CLASSES)), "The index is not in the range of 0-{}".format(len(CLASSES))
        return CLASSES[idx]

    def getTrainSize(self):
        return self._total_train_size

    def getTestSize(self):
        return self._total_test_size

    def preprocessSingleImage(self, fileName, is_training=False):
        """
        In case of testing on a single image
        :return: preprocessed image
        """

        fileName = tf.squeeze(fileName)
        imageStr = tf.io.read_file(fileName)
        image = tf.image.decode_jpeg(imageStr)
        # augment the images
        aug = augment()
        image = aug(image, is_training)
        return image

    def preproc_auto_augment(self, image_np):
        """
        Apply auto-augmentations on the images
        :param image_np: the input single image in Numpy
        :return: the transformed version of the image
        """
        # step 1: convert to PIL image
        # to bring it to [0 - 255] uint8 which is the format pil requires
        image_np = image_np * 255 / np.max(image_np).astype(np.uint8)
        image_pil = Image.fromarray(image_np, 'RGB')

        # step 2: apply augment
        policy = ImageNetPolicy()
        image_pil = policy(image_pil)

        # step 3: convert PIL image back to Numpy ndarray
        image_np = np.array(image_pil)

        return image_np.astype(np.float32)

    def get_keras_ds(self):
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            # rotation_range=40,
            # width_shift_range=0.2,
            # height_shift_range=0.2,
            # rescale=1 / 255,
            # shear_range=0.2,
            # zoom_range=0.2,
            # horizontal_flip=True,
            # fill_mode='nearest',
            preprocessing_function=self.preproc_auto_augment,
            rescale=1.0 / 255.0,
            validation_split=0.1
        )
        # train
        train_generator = train_datagen.flow_from_directory(
            TRAIN_ROOT,
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='training',
        )

        # validation
        validation_generator = train_datagen.flow_from_directory(
            TRAIN_ROOT,
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation'
        )

        # test
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1 / 255,
        )

        test_generator = test_datagen.flow_from_directory(
            TEST_ROOT,
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )

        return train_generator, validation_generator, test_generator

def get_mean_channels():
    """
    Extracting the mean of each channel from the dataset
    :return:
    """
    inData = InputData()
    trainDs, testDs = inData.getDatasets()
    mean = np.zeros((3, 1))
    std = np.zeros((3, 1))
    # First, get the mean of each channel in each batch
    # then, multiply it by the number of samples in that batch
    # then at the end divide it by the total number of samples in the training data
    for image, label in trainDs:
        # print('image.shape {}, label.shape {}'.format(image.shape, label.shape))
        for c in range(3):
            mean[c] += tf.math.reduce_mean(image[:, :, :, c]) * image.shape[0]
            std[c] += tf.math.reduce_std(image[:, :, :, c]) * image.shape[0]
    return mean / inData.getTrainSize(), std / inData.getTrainSize()

if __name__ == "__main__":
    # print(get_mean_channels())
    input_data = InputData()
    train_gen, val_gen, test_gen = input_data.get_keras_ds()