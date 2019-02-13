#!/bin/bash

# activate virtualenv
source /neuroenv/bin/activate

# add module to python path
export PYTHONPATH=/code:$PYTHONPATH

# Force jobs to run on a single thread
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

if [[ -z "$@" ]]; then
    ipython
else
    python "$@"
fi
