#!/bin/bash -eu
# nmtpy-train-2.sh *.conf 0

DEVICE_ID=${@: -1}

for CONF in ${@:1:$#-1}; do
    echo ">" CUDA_VISIBLE_DEVICES=${DEVICE_ID} nmtpy train -C ${CONF}
    CUDA_VISIBLE_DEVICES=${DEVICE_ID} nmtpy train -C ${CONF}
done
