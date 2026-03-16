#!/bin/bash

# Remove the spaces around the equals sign
REAL="./Smiling_final/Smiling_0_1/images_real"
FAKE="./Smiling_final/Smiling_0_1/images_baseline"

# Use quotes around variables to prevent word splitting
#--dims {64,192,768,2048}]

CUDA_VISIBLE_DEVICES=1 python -m pytorch_fid "${REAL}" "${FAKE}" --batch-size 128 --device cuda:0 