#!/bin/bash
mkdir -p ftout

python -m ael.finetune \
    Finetune \
    ./out/best_0.pth \
    ./tests/testdata/systems.dat \
    ./tests/testdata/systems.dat \
    -t ./tests/testdata/systems.dat \
    -d ./tests/testdata \
    -f 0 1 2 \
    -av out/aevc.pth \
    -am out/amap.json \
    -cm out/cmap.json \
    -r 3.5 \
    -b 2 \
    -e 5 \
    --removeHs \
    --plot \
    -o ftout
