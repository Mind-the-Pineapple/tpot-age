#!/bin/bash
singularity run -c \
    -B /home/vagrant/BayOpt/data/:/data \
    -B /home/vagrant/BayOpt/code/:/code \
    $(dirname $0)/preprocessing_fsl.img
