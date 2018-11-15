#!/bin/bash
# arguments order:
#   * $1: dataset
#   * $2: generations
#   * $3: cv
#   * $4: populationsize
#   * $5: resampling factor

# echo "Submitted job for OASIS, 15 generations, 100 cv, 100 populations"
# qsub qsubBayOpt_OASIS.job OASIS 15 100 100
# echo "Submitted job for OASIS, 50 generations, 20 cv, 100 populations"
# qsub qsubBayOpt_OASIS.job OASIS 50 20 100
# echo "Submitted job for OASIS, 15 generations, 20 cv, 100 populations"
# qsub qsubBayOpt_OASIS.job OASIS 15 20 100

# # iterate over different number of generations
# for i in $(seq 0 15 150); do
#     echo "Submitted job for $i Generations"
#     qsub qsubBayOpt_OASIS.job OASIS $i 5 100
# done

# # iterate over different resampling factors
# for i in $(seq 1 1 10); do
#     echo "Submitted job for $i resampling factor"
#     qsub qsubBayOpt_OASIS_with_preprocessing.job OASIS 5 5 100 $i
# done

# iterate over different resampling factors without preprocessing
for i in $(seq 1 1 10); do
    echo "Submitted job for $i resampling factor"
    qsub qsubBayOpt_OASIS_no_preprocessing.job OASIS 5 5 100 $i
done

