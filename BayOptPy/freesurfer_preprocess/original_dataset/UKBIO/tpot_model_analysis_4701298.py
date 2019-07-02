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
This script tests the best model recommened by TPOT for 100 generations, random
seed 42, initial population 1000, mutation rate and cross-validation rate 0.9
and cross-over 0.1
"""

set_publication_style()
# General Settings
#-------------------------------------------------------------------------------
dataset = 'freesurf_combined'
debug = False
resamplefactor = 1
random_seed = 20
save_path = '/code/BayOptPy/tpot/Output/vanilla_combi/100_generations/random_seed_020/'
# Load the clean data for both the UKBIO and the BANC analysis
# This version of the UKBIOBANK dataset contains the same columns as the BANC
# dataset

# Load the saved trained model
tpot = joblib.load(os.path.join(save_path, 'tpot_%s_vanilla_combi_100gen.dump' %dataset))
exported_pipeline = tpot['fitted_pipeline']

# Load the saved validation dataset
project_ukbio_wd, project_data_ukbio, _ = get_paths(debug, dataset)
with open(os.path.join(save_path, 'splitted_dataset.pickle'), 'rb') as handle:
    splitted_dataset = pickle.load(handle)

# Print some results
print('Print MAE - test')
y_predicted_test = exported_pipeline.predict(splitted_dataset['Xtest_scaled'])
mae = mean_absolute_error(splitted_dataset['Ytest'], y_predicted_test)
print(mae)
print('Print MAE - training')
y_predicted_train = exported_pipeline.predict(splitted_dataset['Xtrain_scaled'])
mae_train = mean_absolute_error(splitted_dataset['Ytrain'], y_predicted_train)
print(mae_train)
print('Print MAE - validation')
y_predicted_validation = exported_pipeline.predict(splitted_dataset['Xvalidate_scaled'])
mae_validation = mean_absolute_error(splitted_dataset['Yvalidate'], y_predicted_validation)
print(mae_validation)

# plot predicted vs true for the test
output_path_test = os.path.join(save_path, 'test_predicted_true_age.png')
plot_predicted_vs_true_age(splitted_dataset['Ytest'], y_predicted_test,
                           output_path_test)

# plot predicted vs true for the validation
output_path_val = os.path.join(save_path, 'validation_predicted_true_age.png')
plot_predicted_vs_true_age(splitted_dataset['Yvalidate'],
                           y_predicted_validation, output_path_val)

# Do some statistics. Calculate R2 and the Spearman
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score

rho_val, rho_p_value_val = spearmanr(splitted_dataset['Yvalidate'],
                             y_predicted_validation)
rho_test, rho_p_value_test = spearmanr(splitted_dataset['Ytest'],
                             y_predicted_test)

print('Statistics for the full validation dataset')
print('shape of the dataset: %s' %(splitted_dataset['Yvalidate'].shape,))
print('Rho and p-value: %.4f %.4f' %(rho_val, rho_p_value_val))

r_score = r2_score(splitted_dataset['Yvalidate'], y_predicted_validation)
print('R2 is: %.4f' %r_score)

r_val, r_p_value_val = pearsonr(splitted_dataset['Yvalidate'], y_predicted_validation)
print('R is: %.4f' %r_val)

#-----------------------------------------------------------------------------
# Use just part of the validation dataset
from sklearn.model_selection import train_test_split

X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(
                 splitted_dataset['Xvalidate_scaled'],
                 splitted_dataset['Yvalidate'],
                 test_size=.4, random_state=42)

predicted_subset_val = exported_pipeline.predict(X_train_val)
rho_subset_val, rho_p_value_subset_val = spearmanr(y_train_val,
                                                  predicted_subset_val)

r_subset_val, r_p_value_subset_val = pearsonr(y_train_val,
                                              predicted_subset_val)
print('')
print('Statistics on part of the dataset')
print('shape of the dataset: %s' %(X_train_val.shape,))
print('Rho and p-value: %.4f %.4f' %(rho_subset_val, rho_p_value_subset_val))
print('R is: %.4f' %r_subset_val)

# plot predicted vs true for the test
output_path_test = os.path.join(save_path, 'subset_val_predicted_true_age.png')
plot_predicted_vs_true_age(y_train_val, predicted_subset_val,
                           output_path_test)

#----------------------------------------------------------------------------
# Retrain the best model on the validation dataset

