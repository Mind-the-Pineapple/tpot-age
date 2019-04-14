import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import os
import joblib

# Load the analysis results
path = '/code/BayOptPy/tpot/vanilla_combi/100_generations/random_seed_010/tpot_BANC_freesurf_vanilla_combi_100gen.dump'
results = joblib.load(path)

# Define the list of possible models
algorithms_list = ['GaussianProcessRegressor', 'RVR', 'LinerSVR',
          'RandomForestRegressor', 'KNeighborsRegressors',
          'LinearRegression', 'Ridge','ElasticNetCV',
          'ExtraTreesRegressor', 'LassoLarsCV',
          'DecisionTreeRegressor']
algorithms_dic = dict.fromkeys(results['evaluated_individuals'].keys(), {})
# Append empty dictionary with the list of models to each generation
for generation in results['evaluated_individuals']:
    algorithms_dic[generation] = dict.fromkeys(algorithms_list, 0)

# Iterate over all the dictionary keys from the dictionary with the TPOT
# analysed model and count the ocurrence of each one of the algorithms of
# interest
for generation in results['evaluated_individuals']:
    for algorithm in algorithms_dic[generation]:
        for tpot_pipeline in results['evaluated_individuals'][generation]:
            # By using '( we cound the algorithm only once and do not
            # care about its hyperparameters definitions'
            algorithms_dic[generation][algorithm] += tpot_pipeline.count(algorithm + '(')

import pdb
pdb.set_trace()
