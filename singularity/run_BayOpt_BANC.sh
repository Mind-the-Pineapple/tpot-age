#!/bin/bash

# check if the /data folder exists, and determine the data location accordingly
if [ -d /data ]; then # Cluster path
    datapath=/data/project/brainage
elif [ -d $HOME/myHome ]; then # vagrant path
    datapath=$HOME/myHome
else # best path
    datapath=$HOME
fi
echo "Data path is:"

echo $datapath
singularity run -c \
            -B ~/BayOpt/:\code \
            -B ~/BayOpt/singularity:\sing \
            -B $datapath:/data/NaN \
            -B /opt/sge/:/opt/sge \
            $(dirname $0)/BayOpt.img
