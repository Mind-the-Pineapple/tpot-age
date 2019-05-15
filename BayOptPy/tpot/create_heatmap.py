import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import os
import joblib

# This script creates the heatplot for one single type of analysis that is
# specified on the path

# Load the analysis results
path = '/code/BayOptPy/tpot/vanilla_combi/100_generations/random_seed_010/tpot_BANC_freesurf_vanilla_combi_100gen.dump'
results = joblib.load(path)

# Define the list of possible models
algorithms_list = ['GaussianProcessRegressor', 'RVR', 'LinearSVR',
          'RandomForestRegressor', 'KNeighborsRegressor',
          'LinearRegression', 'Ridge','ElasticNetCV',
          'ExtraTreesRegressor', 'LassoLarsCV',
          'DecisionTreeRegressor']

df = pd.DataFrame(columns=algorithms_list, index=results['evaluated_individuals'].keys())

# Iterate over all the dictionary keys from the dictionary with the TPOT
# analysed model and count the ocurrence of each one of the algorithms of
# interest
for generation in results['evaluated_individuals']:
    algorithms_dic = dict.fromkeys(algorithms_list, 0)
    for algorithm in algorithms_list:
        for tpot_pipeline in results['evaluated_individuals'][generation]:
            # By using '( we cound the algorithm only once and do not
            # care about its hyperparameters definitions'
            algorithms_dic[algorithm] += tpot_pipeline.count(algorithm + '(')
    # Append the count of algorithms to the dataframe
    df.loc[generation] = pd.Series(algorithms_dic)

# Create heatmap
df2 = df.transpose()
plt.figure(figsize=(8,8))
sns.heatmap(df2, cmap='YlGnBu')
plt.xlabel('Generations')
plt.tight_layout()
plt.savefig('/code/BayOptPy/tpot/heatmap.png')
