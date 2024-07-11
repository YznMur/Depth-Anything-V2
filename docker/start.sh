#!/bin/bash
cd "$(dirname "$0")"
cd ..

workspace_dir=$PWD
 
if ["$(docker ps -aq -f status=exited -f name=anydepth)" ]; then
    docker rm anydepth
fi

docker run -it -d --rm \
    --gpus all \
    --net host \
    -e "NVIDIA_DRIVER_CAPABILITIES=all" \
    -e "DISPLAY" \
    -e "QT_X11_NO_MITSHM=1" \
    --shm-size="45g" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --name anydepth \
    -v $workspace_dir/:/home/trainer/anydepth/:rw \
    -v /media/yazan/Samsung_SSD/mapping_workspace/anydepth_output:/home/trainer/anydepth/output/:rw \
    x64/anydepth:latest
