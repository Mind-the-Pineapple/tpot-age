#!/bin/bash
# check if the /data folder exists, and determine the data location accordingly
if [ -d /data ]; then #cluster path
        datapath=/data/project/brainage
elif [ -d $HOME/myHome ]; then # vagrant path
        datapath=$HOME/myHome
else # beast path
        datapath=$HOME
fi
echo "Data path is:"

# check if the /data folder exists, and determine the data location accordingly
singularity exec -c \
            -B $HOME/BayOpt/:\code \
            -B $HOME/BayOpt/singularity:\sing \
            -B $datapath:/data \
            $(dirname $0)/BayOpt.img "$@"
