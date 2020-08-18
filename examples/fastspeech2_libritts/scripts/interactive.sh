#!/bin/bash
docker run --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm --ipc=host -p 8888:8888 -v $PWD:/workspace/tts/ tftts bash
