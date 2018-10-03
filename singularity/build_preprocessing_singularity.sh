#!/bin/bash

set -e

# If it exists remove old image
if [ -f preprocessing_fsl.img ]; then
    rm preprocessingSingularity preprocessing_fsl.img
    echo 'Remove old files'
fi

mv preprocessing5 preprocessingSingularity
echo 'rename preprocessing5 into preprocessingSingularity'

singularity image.create preprocessing_fsl.img
echo 'created singularity image'

sudo singularity build preprocessing_fsl.img preprocessingSingularity
echo 'Linked image to recipe'
