#!/bin/bash --login
# How much memory needed *per core*
#$ -l h_vmem=1G

# Which operating system to use
#$ -l cns_os=el7

# Where to save the STDOUT and STDERR
#$ -o ~/BayOpt/singularity/logs/tests
#$ -e ~/BayOpt/singularity/logs/tests

/home/k1506210/BayOpt/singularity/exec_BayOpt_BANC.sh /sing/neuroenvpy.sh \
/code/BayOptPy/tpot/brain_age_analysis.py -raw False -model classification \
-predicted_attribute Prospective_memory -dataset UKBIO_freesurf \
-cv 5 -generations 3 -population_size 5 -offspring_size 5 -config_dict None \
-njobs 1 -random_seed 999 -analysis vanilla_combi -mutation_rate 0.9 \
-crossover_rate 0.1
