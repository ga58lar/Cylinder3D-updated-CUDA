#!/bin/bash

# Check correct call of script
if [ $# -ne 1 ]; then
    echo "Specify data-mount path!"
    exit 1
else
    MNT=$1
fi
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )" 
tag=latest 

docker run --rm -it \
    --network=host \
    -v /dev/shm:/dev/shm \
    -e DISPLAY=$DISPLAY \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    --gpus all \
    -v /tmp/.X11-unix/:/tmp/.X11-unix/ \
    --privileged \
    -v $MNT:/data \
    -v $SCRIPT_DIR/..:/dev_ws \
    --runtime=nvidia \
    cylinder3d:$tag \
    bash
