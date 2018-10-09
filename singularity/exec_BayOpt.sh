#!/bin/bash
singularity exec -c \
            -B /home/vagrant/BayOpt/:\BayOpt \
            $(dirname $0)/BayOpt.img "$@"
