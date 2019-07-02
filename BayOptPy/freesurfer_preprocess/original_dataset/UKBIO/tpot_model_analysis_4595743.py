import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
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
                                      set_publication_style)

"""
BANC + TPOT dataset
This script tests the best model recommened by TPOT for 100 generations, random
seed 42, initial population 1000, mutation rate and cross-validation rate 0.9
and cross-over 0.1
"""

set_publication_style()
# General Settings
#-------------------------------------------------------------------------------
debug = False
resamplefactor = 1
random_seed = 42
save_path = '/code/BayOptPy/tpot/Output/random_seed/100_generations/random_seed_042/'
# Load the clean data for both the UKBIO and the BANC analysis
# This version of the UKBIOBANK dataset contains the same columns as the BANC
# dataset
project_ukbio_wd, project_data_ukbio, _ = get_paths(debug, 'UKBIO_freesurf')
_, _, df_ukbio =  \
             get_data(project_data_ukbio, 'UKBIO_freesurf', debug,
                      project_ukbio_wd, resamplefactor, raw=False, analysis=None)
df_ukbio = df_ukbio.set_index('id')
# Drop the last column that corresponds the name of the dataset
df_ukbio = df_ukbio.drop('dataset', axis=1)

project_banc_wd, project_banc_data, _ = get_paths(debug,'BANC_freesurf')
demographics_banc,__, df_banc = get_data(project_banc_data,
                                                'BANC_freesurf',
                                         debug, project_banc_wd,
                                         resamplefactor, raw=False,
                                         analysis=None)
# Drop the last column that corresponds the name of the dataset
df_banc = df_banc.drop('dataset', axis=1)

# Get age for the BIOBANK dataset
age_UKBIO = pd.read_csv(os.path.join(project_data_ukbio, 'original_dataset',
'UKBIO','UKB_FS_age_sex.csv'))
targetAttribute_ukbio = np.array(age_UKBIO['age'])

#-------------------------------------------------------------------------------
# Train the model with BANC
#-------------------------------------------------------------------------------
data_banc = df_banc.values
data_ukbio = df_ukbio.values
# Find age for the BANC dataset
targetAttribute_banc = np.array(demographics_banc['age'])

# Add a few of the BIOBANK Dataset into the training set
Xtrain_ukbio, Xtest_ukbio, Ytrain_ukbio, Ytest_ukbio = train_test_split(data_ukbio, targetAttribute_ukbio,
                                                test_size=.25,
                                                random_state=random_seed)

Xtrain_banc, Xtest_banc, Ytrain_banc, Ytest_banc = train_test_split(data_banc, targetAttribute_banc,
                                                test_size=.25,
                                                random_state=random_seed)

# Keep part of the BIOBANK outsite for validaion purposes
# Select every second subject to be part of the validation set
Yvalidation_ukbio = Ytest_ukbio[0:-1:2]
Xvalidation_ukbio = Xtest_ukbio[0:-1:2]
# Concatenate both datasets
Xtrain = np.concatenate((Xtrain_banc, Xtrain_ukbio[1:-1:2]), axis=0)
Xtest = np.concatenate((Xtest_banc, Xtest_ukbio[1:-1:2]), axis=0)
Ytrain = np.concatenate((Ytrain_banc, Ytrain_ukbio[1:-1:2]), axis=0)
Ytest = np.concatenate((Ytest_banc, Ytest_ukbio[1:-1:2]), axis=0)

print('Divided BANC dataset into test and training')
print('Check train test split sizes')
print('X_train: ' + str(Xtrain.shape))
print('X_test: '  + str(Xtest.shape))
print('Y_train: ' + str(Ytrain.shape))
print('Y_test: '  + str(Ytest.shape))
print('X_valitation ' + str(Xvalidation_ukbio.shape))
print('Y_test: ' + str(Yvalidation_ukbio.shape))

# Best pipeline recommended by TPOT
exported_pipeline = make_pipeline(
                           StackingEstimator(estimator=Ridge(alpha=100.0,
                                                             random_state=42)),
                           MinMaxScaler(),
                           StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.8,
                                                                    random_state=42,
                                                                    tol=0.1)),
                           StackingEstimator(estimator=RandomForestRegressor(bootstrap=True,
                                                                             max_features=0.45,
                                                                             min_samples_leaf=5,
                                                                             min_samples_split=15,
                                                                             n_estimators=100,
                                                                             random_state=42)),
                           RandomForestRegressor(bootstrap=True,
                                                 max_features=0.6500000000000001,
                                                 min_samples_leaf=2,
                                                 min_samples_split=12,
                                                 n_estimators=100,
                                                 random_state=42)
                                 )

exported_pipeline.fit(Xtrain, Ytrain)
print('Print MAE - test')
y_predicted = exported_pipeline.predict(Xtest)
mae = mean_absolute_error(Ytest, y_predicted)
print(mae)
print('Print MAE - training')
y_predicted_train = exported_pipeline.predict(Xtrain)
mae_train = mean_absolute_error(Ytrain, y_predicted_train)
print(mae_train)
print('Print MAE - validation')
y_predicted_validation = exported_pipeline.predict(Xvalidation_ukbio)
mae_validation = mean_absolute_error(Yvalidation_ukbio, y_predicted_validation)
print(mae_validation)

# plot predicted vs true for the test
fig = plt.figure()
plt.scatter(Ytest, y_predicted)
plt.ylabel('Predicted Age')
plt.xlabel('True Age')
plt.savefig(os.path.join(save_path, 'test_predicted_true_age.png'))
plt.close()

# plot predicted vs true for the validation
fig = plt.figure()
plt.scatter(Yvalidation_ukbio, y_predicted_validation)
plt.ylabel('Predicted Age')
plt.xlabel('True Age')
plt.savefig(os.path.join(save_path, 'validation_predicted_true_age.png'))
plt.close()
