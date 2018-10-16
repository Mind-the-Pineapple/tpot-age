#!/bin/bash

# activate virtualenv
source /neuroenv/bin/activate

# add module to python path
export PYTHONPATH=/code:$PYTHONPATH

if [[ -z "$@" ]]; then
    ipython
else
    python "$@"
fi
