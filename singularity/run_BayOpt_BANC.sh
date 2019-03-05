#!/bin/bash

# check if the /data folder exists, and determine the data location accordingly
if [ -d /data ]; then # Cluster path
    datapath=/data/project/brainage
elif [ -d $HOME/myHome ]; then # vagrant path
    datapath=$HOME/myHome/NaN
else # best path
    datapath=$HOME/NaN
fi
echo "Data path is:"

echo $datapath
/opt/singularity/bin/singularity run -c \
            -B ~/BayOpt/:\code \
            -B ~/BayOpt/singularity:\sing \
            -B $datapath:/data/NaN \
            $(dirname $0)/BayOpt.img
