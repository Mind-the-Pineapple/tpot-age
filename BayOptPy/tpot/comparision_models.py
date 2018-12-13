import os
import numpy as np
from joblib import load
from matplotlib.pylab import plt

dataset = 'BANC_freesurf'
tpot_dict = 'gpr'
project_wd = os.getcwd()
saved_tpot = load(os.path.join(project_wd, 'tpot_%s_%s.dump' %(dataset,
                                                               tpot_dict)))

# Get the internal cross validation scores
cv_scores = [saved_tpot['evaluated_individuals_'][model]['internal_cv_score']
             for model in saved_tpot['evaluated_individuals_'].keys()]
model_names = [model for model in saved_tpot['evaluated_individuals_'].keys()]
# Just a quick solution to plot something, but you should plot histograms
ind = np.arange(len(cv_scores))
plt.bar(ind, cv_scores)
plt.ylabel('MAE')
plt.xticks(ind)
plt.show()

plt.figure()
plt.scatter(cv_scores)
plt.show()
