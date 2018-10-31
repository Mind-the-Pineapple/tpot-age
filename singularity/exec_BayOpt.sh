#!/bin/bash

# check if the /data folder exists, and determine the data location accordingly
if [ -d /data ]; then
    datapath=/data/group/neurauto/BayOpt/data
else
    datapath=$HOME/BayOpt/data
fi
echo "Data path is:"
echo $datapath

singularity exec -c \
            -B $HOME/BayOpt/:\code \
            -B $HOME/BayOpt/singularity:\sing \
            -B $datapath:/data \
            $(dirname $0)/BayOpt.img "$@"
