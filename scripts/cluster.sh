#!/bin/sh
ARCH=$1

if [ $# -ne 1 ]
  then
    echo "Arguments error: <ARCH>"
    exit 1
fi

python -u datasets/cluster.py -d pitts -a ${ARCH} -b 64 --width 640 --height 480 \
  --resume logs/pitts30k-vgg16/model_best.pth.tar
