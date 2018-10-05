#!/bin/sh

set -e

# Singularity recipe.
generate() {
    docker run --rm kaczmarj/neurodocker:0.4.1 generate "$1" \
        --base=neurodebian:stretch-non-free \
        --pkg-manager=apt \
        --install fsl vim git coreutils\
        --add-to-entrypoint='source /etc/fsl/fsl.sh' \
        --miniconda create_env=neuro \
                    conda_install='python=3.6 numpy=1.15.2 pandas=0.23.4 nipype=1.1.3
                                 scipy=1.1.0 scikit-learn=0.19.2 ipython=6.5.0' \
                    activate=True \
        --user=neuro \
        --workdir='/home/neuro'
}

generate singularity > Singularity
