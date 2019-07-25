import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import joblib
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import ElasticNetCV
from tpot.builtins import StackingEstimator

from BayOptPy.helperfunctions import (get_paths, get_data,
                                      drop_missing_features,
                                      set_publication_style,
                                      plot_predicted_vs_true_age)

"""
BANC + TPOT dataset
This script tests the best model recommened by TPOT for 100 generations for
multiple random seeds
"""

set_publication_style()
# General Settings
#-------------------------------------------------------------------------------
dataset = 'freesurf_combined'
debug = False
resamplefactor = 1
random_seeds = np.arange(0, 110, 10)

def tpot_confusion_matrix(random_seed, save_path):
    save_path = os.path.join(save_path, 'random_seed_%03d' %random_seed)
    print('Random seed: %03d' %random_seed)
    # Load the clean data for both the UKBIO and the BANC analysis
    # This version of the UKBIOBANK dataset contains the same columns as the BANC
    # dataset

    # Load the saved trained model
    tpot = joblib.load(os.path.join(save_path, 'tpot_%s_vanilla_classification_100gen.dump' %dataset))
    tpot_con_test = tpot['confusion_matrix_test']
    tpot_con_val = tpot['confusion_matrix_validatate']
    return tpot_con_test, tpot_con_val

save_path = '/code/BayOptPy/tpot_classification/Output/vanilla_combi/age/100_generations/'
confusion_matrix_test_all = []
confusion_matrix_validation_all = []
tpot_best_models_all = []
for random_seed in random_seeds:
    con_matrix_test, con_matrix_validation = tpot_confusion_matrix(random_seed, save_path)
    confusion_matrix_test_all.append(con_matrix_test)
    confusion_matrix_validation_all.append(con_matrix_validation)

print('Mean and std for test data')
print(np.mean(confusion_matrix_test_all, axis=0),
      np.std(confusion_matrix_test_all, axis=0))
print('Mean and std for validation data')
print(np.mean(confusion_matrix_validation_all, axis=0),
      np.std(confusion_matrix_validation_all, axis=0))

results = {'confusion_matrix_test': confusion_matrix_test_all,
           'confusion_matrix_validation': confusion_matrix_validation_all,
           }
with open(os.path.join(save_path, 'tpot_all_seeds.pckl'), 'wb') as handle:
    pickle.dump(results, handle)




