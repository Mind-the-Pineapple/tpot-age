#!/bin/bash

set -e

echo "---------------------------"
echo "Removing previous container"
echo "---------------------------"
sudo rm -rf BayOpt.img

echo "----------------------------------"
echo "Linking image to singularity file"
echo "----------------------------------"
sudo singularity build --sandbox BayOpt.img Singularity

