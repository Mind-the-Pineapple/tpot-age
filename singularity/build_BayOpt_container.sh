#!/bin/bash

set -e

if [ -d BayOpt.img ]; then
    sudo rm -rf BayOpt.img
    echo 'Remove old files'
fi

sudo singularity build --sandbox BayOpt.img BayOptSingularity
echo 'Linked image to recipe'

