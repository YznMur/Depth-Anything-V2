#!/bin/bash
docker exec -it anydepth \
    /bin/bash -c "
    cd /home/trainer/anydepth;
    nvidia-smi;
    /bin/bash"