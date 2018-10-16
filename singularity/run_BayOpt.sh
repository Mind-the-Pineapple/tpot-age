#!/bin/bash
singularity run -c \
            -B ~/BayOpt/:\code \
            -B ~/BayOpt/singularity:\sing \
            -B ~/BayOpt/data:\data \
            $(dirname $0)/BayOpt.img
