#!/bin/bash
singularity run -c \
            -B /home/vagrant/BayOpt/:\BayOpt \
            $(dirname $0)/BayOpt.img
