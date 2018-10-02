#!/bin/bash

set -e

echo "---------------------------"
echo "Removing previous container"
echo "---------------------------"
sudo rm -rf BayOpt.img

echo "---------------------------"
echo " Creating empty image"
echo "---------------------------"
singularity image.create BayOpt.img

echo "----------------------------------"
echo "Linking image to singularity file"
echo "----------------------------------"
sudo singularity build BayOpt.img Singularity

