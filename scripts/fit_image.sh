#!/bin/bash

# Set the CUDA device (change the index as needed)
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0; change to "1,2" for multiple GPUs

# Run the Python script
python fit_image.py \
