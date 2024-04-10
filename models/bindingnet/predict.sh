#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python -m ael.predict \
    bindingnet \
    ./data/plb/merck_nhm.dat \
    ./experiments/bindingnet/bindingnet_pdbbind-2019_merck_schrodinger_nhm_randomsplit_consensus_ecfp/best_0.pth ./experiments/bindingnet/bindingnet_pdbbind-2019_merck_schrodinger_nhm_randomsplit_consensus_ecfp/best_1.pth ./experiments/bindingnet/bindingnet_pdbbind-2019_merck_schrodinger_nhm_randomsplit_consensus_ecfp/best_2.pth ./experiments/bindingnet/bindingnet_pdbbind-2019_merck_schrodinger_nhm_randomsplit_consensus_ecfp/best_3.pth ./experiments/bindingnet/bindingnet_pdbbind-2019_merck_schrodinger_nhm_randomsplit_consensus_ecfp/best_4.pth \
    -d /biggin/t001/bioc1805/Git/fep-merck/data/ \
    -e ./experiments/bindingnet/bindingnet_pdbbind-2019_merck_schrodinger_nhm_randomsplit_consensus_ecfp/aevc.pth \
    -am ./experiments/bindingnet/bindingnet_pdbbind-2019_merck_schrodinger_nhm_randomsplit_consensus_ecfp/amap.json \
    -cm ./experiments/bindingnet/bindingnet_pdbbind-2019_merck_schrodinger_nhm_randomsplit_consensus_ecfp/cmap.json \
    -r 3.5 \
    -b 64 \
    -o ./experiments/bindingnet/bindingnet_pdbbind-2019_merck_schrodinger_nhm_randomsplit_consensus_ecfp \

mv ./experiments/bindingnet/bindingnet_pdbbind-2019_merck_schrodinger_nhm_randomsplit_consensus_ecfp/predict.csv ./experiments/bindingnet/bindingnet_pdbbind-2019_merck_schrodinger_nhm_randomsplit_consensus_ecfp/test_merck.csv

python -m ael.predict \
    bindingnet \
    ./data/plb/schrodinger_nhm.dat \
    ./experiments/bindingnet/bindingnet_pdbbind-2019_merck_schrodinger_nhm_randomsplit_consensus_ecfp/best_0.pth ./experiments/bindingnet/bindingnet_pdbbind-2019_merck_schrodinger_nhm_randomsplit_consensus_ecfp/best_1.pth ./experiments/bindingnet/bindingnet_pdbbind-2019_merck_schrodinger_nhm_randomsplit_consensus_ecfp/best_2.pth ./experiments/bindingnet/bindingnet_pdbbind-2019_merck_schrodinger_nhm_randomsplit_consensus_ecfp/best_3.pth ./experiments/bindingnet/bindingnet_pdbbind-2019_merck_schrodinger_nhm_randomsplit_consensus_ecfp/best_4.pth \
    -d /biggin/t001/bioc1805/Git/fep-schrodinger/data/ \
    -e ./experiments/bindingnet/bindingnet_pdbbind-2019_merck_schrodinger_nhm_randomsplit_consensus_ecfp/aevc.pth \
    -am ./experiments/bindingnet/bindingnet_pdbbind-2019_merck_schrodinger_nhm_randomsplit_consensus_ecfp/amap.json \
    -cm ./experiments/bindingnet/bindingnet_pdbbind-2019_merck_schrodinger_nhm_randomsplit_consensus_ecfp/cmap.json \
    -r 3.5 \
    -b 64 \
    -o ./experiments/bindingnet/bindingnet_pdbbind-2019_merck_schrodinger_nhm_randomsplit_consensus_ecfp \

mv ./experiments/bindingnet/bindingnet_pdbbind-2019_merck_schrodinger_nhm_randomsplit_consensus_ecfp/predict.csv ./experiments/bindingnet/bindingnet_pdbbind-2019_merck_schrodinger_nhm_randomsplit_consensus_ecfp/test_schrodinger.csv
