#!/bin/bash

rm -rf checkpoints/*
rm -rf logs/*

docker run --runtime=nvidia -it --rm -v $PWD:/tmp -w /tmp my-docker python ./fishFinder.py
