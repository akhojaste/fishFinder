#!/bin/bash

echo "Running the training script, current working directory:"
echo $PWD

echo "************************************************************"

echo "Removing the checkpoints and logs directory"
rm -rf checkpoints/*
rm -rf logs/*

echo "************************************************************"

docker run --runtime=nvidia -it --rm -v $PWD:/tmp -w /tmp my-docker python fishFinder.py
