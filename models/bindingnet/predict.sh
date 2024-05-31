#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python -m ael.predict \
    experiment_name \
    /path/to/systems.dat \
    ./models/bindingnet/best_0.pth ./models/bindingnet/best_1.pth ./models/bindingnet/best_2.pth ./models/bindingnet/best_3.pth ./models/bindingnet/best_4.pth \
    -d /path/to/structure_files \
    -e ./models/bindingnet/aevc.pth \
    -am ./models/bindingnet/amap.json \
    -cm ./models/bindingnet/cmap.json \
    -r 3.5 \
    -b 64 \
    -o /path/to/output_dir \
