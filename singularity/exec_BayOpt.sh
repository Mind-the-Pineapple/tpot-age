#!/bin/bash
singularity exec -c \
            -B /home/vagrant/BayOpt/:\code \
            -B /home/vagrant/BayOpt/singularity:\sing \
            -B /home/vagrant/BayOpt/data:\data \
            $(dirname $0)/BayOpt.img "$@"
