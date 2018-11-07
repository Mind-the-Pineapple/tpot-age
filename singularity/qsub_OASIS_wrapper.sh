#!/bin/bash
# arguments order:
#   * $1: dataset
#   * $2: generations
#   * $3: cv
#   * $4: populationsize

echo "Submitted job for OASIS, 15 generations, 100 cv, 100 populations"
qsub qsubBayOpt_OASIS.job OASIS 15 100 100
echo "Submitted job for OASIS, 50 generations, 20 cv, 100 populations"
qsub qsubBayOpt_OASIS.job OASIS 50 20 100
echo "Submitted job for OASIS, 15 generations, 20 cv, 100 populations"
qsub qsubBayOpt_OASIS.job OASIS 15 20 100


