# Fishfinder

During this summer I have been trying fishing for the first time in my life. I have really enjoyed catching all the (small!!) fishes I caught. As a newbie, I was always struggling to know the type of fish I was catching. That's why I have been doing this project to classify the fish I take using the pictures of my cellphone camera. 

## Classification
For classification, I have been doing transfer-learning on Mobilenet_v2 pretrained on Imagennet. 

## Dataset
I have downloaded all the fish images from Google Images. 

## Technology stack
I have used tensorflow 2.0 for this project. Also for deployment on cellphone I have used the tensorflow TFLite model deployment. For now, I have used the out of the box app that comes with TFLite, however, plan is to write my own app for this too.
