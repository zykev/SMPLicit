#!/bin/bash

# Set the CUDA device (change the index as needed)
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0; change to "1,2" for multiple GPUs

# Run the Python script
python dress4d_process.py \
    --subj 00123 \
    