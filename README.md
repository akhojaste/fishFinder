# Fishfinder

During this summer I have been trying fishing for the first time in my life. As a newbie, I always struggled to know the type of the fish I was catching. That's why I have been doing this project to classify the fish I take using the cellphone camera. 

## Classification
I have done transfer-learning on Mobilenet_v2 pretrained on Imagennet. 

## Dataset
I have downloaded all the fish images from Google Images. So far, only 4 fish types are supported. Carp, Largemouth_bass, Pike and Crappie. 

## Technology stack
I have used tensorflow 2.0 and Keras API for this project. Also for deployment on cellphone I have used the tensorflow TFLite model deployment.

# How to run

## Requirements

* Python 3.6
* docker 19.03

The Dockerfile will load a gpu version of tensorflow 2.0 and installs the necessary python libraries into the image. To start training run the blow command:

* sh ./scripts/run.sh

To get the tensorboard run the below command:

* sh ./scripts/tensorboard.sh

To see the nvidia-smi results on the docker image run:

* sh ./scripts/nvidia.sh


