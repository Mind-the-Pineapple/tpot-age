import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
sns.set()

from BayOptPy.helperfunctions import (get_mae_for_all_generations,
                                      get_all_random_seed_paths,
                                      set_publication_style)


parser = argparse.ArgumentParser()
parser.add_argument('-dataset',
                   dest='dataset',
                   help='Specify which dataset to use',
                   choices=['OASIS', 'BANC',
                            'BANC_freesurf']
                   )
parser.add_argument('-generations',
                    dest='generations',
                    help='Specify number of generations to use',
                    type=int,
                    required=True
                    )
parser.add_argument('-analysis_list',
                    dest='analysis_list',
                    help='Specify the list of analysis',
                    required=True,
                    nargs='+',
                   )
parser.add_argument('-population_size',
                    dest='population_size',
                    help='Specify population size to use',
                    type=int,
                    default=100 # use the same default as TPOT default population value
                    )
parser.add_argument('-random_seed',
                    dest='random_seed',
                    help='Specify random seed to use',
                    required=True,
                    type=int
                   ),

args = parser.parse_args()

debug=False
# Define mutation and cross over (The values below are the default TPOT)
mutation = .9
crossover = .1

set_publication_style()
plt.figure()
for analysis in args.analysis_list:
    print('Analysis Type: %s' %analysis)
    tpot_path = get_all_random_seed_paths(analysis, args.generations,
                                          args.population_size,
                                          debug, mutation, crossover)

    all_mae_test_set, all_mae_train_set, pipeline_complexity = \
            get_mae_for_all_generations(args.dataset,
                                        args.random_seed,
                                        args.generations,
                                        analysis, tpot_path)

    # Plot complexity as a bar plot for each generation
    plt.plot(range(len(pipeline_complexity)), pipeline_complexity, marker='o',
                label=analysis)
plt.xlabel('Generation')
plt.ylabel('Pipeline complexity')
plt.legend()
plt.ylim(0, 10)
plt.savefig(os.path.join(os.path.dirname(tpot_path), 'pipeline_complexity.png'))
