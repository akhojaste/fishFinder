# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 09:49:21 2019

@author: amir
"""
import tensorflow as tf
import os
from os import walk

##-----------------------------
ROOT = "D:\Fish\DATA"
CLASSES = ['carp', 'largemouth_bass', 'pike', 'crappie']
NUM_CLASSES = len(CLASSES)
IMAGE_SIZE = 160 #MobileNetV2
CROP_LENGTH = 30
##-----------------------------


class InputData:
    
    def __init__(self):
        """
        Arguments:
            None
        Returns:
            None
        """
        self.trainTestSplit = 0.2
        self.batchSize = 32
        self._createTrainTestDataset()
        
        
    def _getFileNames(self, root):

        fileNames = []
        labels = []
        
        
        for idx , label in enumerate(CLASSES):
            for (dirpath, dirnames, names) in walk(os.path.join(ROOT, label)):
                for file in names:                
                    fileNames.append(os.path.join(dirpath, file))
                    labels.append(idx) #One hot it later
            
        return fileNames, labels
        

    def _createTrainTestDataset(self):
        
        self.fileNames, self.labels = self._getFileNames(ROOT)

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

    def preprocessSingleImage(self, fileName, is_training=False):
        """
        In case of testing on a single image
        :return: preprocessed image
        """

        fileName = tf.squeeze(fileName)
        imageStr = tf.io.read_file(fileName)
        image = tf.image.decode_jpeg(imageStr)

        if is_training:
            # Augmentations
            # image_shape = image.shape
            # if image_shape[0] == None or image_shape[1]==None:
            #     print('*' * 20)
            #     tf.print("fileName", [fileName])
            #     print(image_shape)
            #     print('*' * 20)
            #     raise Exception('Alioooooooooooooooooooooooo')
            # crop_size = [tf.subtract(image_shape[0], CROP_LENGTH), tf.subtract(image_shape[1], CROP_LENGTH), 3]
            # image = tf.image.random_crop(image, size=crop_size)

            image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
            image = tf.image.random_crop(image, [IMAGE_SIZE - CROP_LENGTH, IMAGE_SIZE - CROP_LENGTH, 3])
            image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
            image /= 255.0  # [0-1]
            image = tf.image.random_flip_left_right(image)
            return image

        else:
            image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
            image /= 255.0  # [0-1]
            return image


if __name__ == "__main__":

    inData = InputData()
    trainDs, testDs = inData.getDatasets()

    # dataset has not make_one_shot_iterator anymore in tf 2.0!!!
    itr = tf.compat.v1.data.make_one_shot_iterator(trainDs)
    image, label = itr.get_next()

    for itr in range(0, 2):
        print('image.shape {}, label.shape {}'.format(image.shape, label.shape))
        print(label)
