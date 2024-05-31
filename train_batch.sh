#!/bin/bash

where="/biggin/t001/bioc1805/Git/aescore/experiments/bindingnet/bindingnet_pdbbind-2019_nhm_undersampled"

for folder in "$where"/*; do
    if [ -d "$folder" ]; then
        echo "Processing folder: $folder"
        
        if [ -f "$folder/train.sh" ]; then

            if [ -f "$folder/test.csv" ]; then
                echo "Warning: test.csv already exists in $folder. Skipping."
                continue
            fi

            chmod +x "$folder/train.sh"
            "$folder/train.sh"

            sleep 30
        else
            echo "Warning: train.sh not found in $folder. Skipping."
        fi
    fi
done
