#!/bin/bash

# check if the /data folder exists, and determine the data location accordingly
singularity exec -c \
            -B $HOME/BayOpt/:\code \
            -B $HOME/BayOpt/singularity:\sing \
            -B $datapath:/data \
            $(dirname $0)/BayOpt.img "$@"
