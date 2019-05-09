import numpy as np
import pandas as pd
import os
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.metrics import mean_absolute_error
from tpot.builtins import StackingEstimator

from BayOptPy.helperfunctions import get_paths, get_data, drop_missing_features

"""
This script tests the best model recommened by TPOT for 100 generations, random
seed 20, initial population 1000, mutation rate and cross-validation ratexxx
"""

# General Settings
#-------------------------------------------------------------------------------
debug = False
resamplefactor = 1
random_seed = 20
project_ukbio_wd, project_data_ukbio, _ = get_paths(debug, 'UKBIO_freesurf')
# demographics_ukbio, imgs_ukbio, data_ukbio, freesurfer_df_ukbio =  \
#             get_data(project_data_ukbio, 'UKBIO_freesurf', debug, project_ukbio_wd, resamplefactor)

# Retrain the model using the BANC dataset
#-------------------------------------------------------------------------------
project_banc_wd, project_banc_data, _ = get_paths(debug,'BANC_freesurf')
demographics_banc,__, df_banc = get_data(project_banc_data,
                                                'BANC_freesurf',
                                                debug, project_banc_wd,
                                                resamplefactor)
# Drop missing features from BIOBANK. So that BIOBANK and BANC have the same
# number of features
df_banc = drop_missing_features(df_banc)
data_banc =  df_banc.values
# Find age for the BANC dataset
targetAttribute_banc = np.array(demographics_banc['Age'])
Xtrain, Xtest, Ytrain, Ytest = train_test_split(data_banc, targetAttribute_banc,
                                                test_size=.25,
                                                random_state=random_seed)
print('Divided BANC dataset into test and training')
print('Check train test split sizes')
print('X_train: ' + str(Xtrain.shape))
print('X_test: '  + str(Xtest.shape))
print('Y_train: ' + str(Ytrain.shape))
print('Y_test: '  + str(Ytest.shape))

# Best pipeline recommended by TPOT
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=Ridge(alpha=10.0, random_state=42)),
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=True, max_features=0.7500000000000001, min_samples_leaf=4, min_samples_split=4, n_estimators=100, random_state=42)),
    ExtraTreesRegressor(bootstrap=False, max_features=0.9000000000000001, min_samples_leaf=3, min_samples_split=2, n_estimators=100, random_state=42)
)

exported_pipeline.fit(Xtrain, Ytrain)
y_predicted_banc = exported_pipeline.predict(Xtest)
mae_banc = mean_absolute_error(Ytest, y_predicted_banc)

# Get the same features for the BIOBANK dataset
#-------------------------------------------------------------------------------
df_UKBIO = pd.read_csv(os.path.join(project_data_ukbio, 'UKB_FS_age_sex.csv'))
features = np.array(df_UKBIO['age'])

# This version of the UKBIOBANK dataset contains the same columns as the BANC
# dataset
df_UKBIO = pd.read_csv(os.path.join(project_data_ukbio, 'UKB_10k_FS_4844_adapted.csv'))
df_UKBIO = df_UKBIO.set_index('ID')
# Drop the last column that corresponds the name of the dataset
df_UKBIO = df_UKBIO.drop('dataset', axis=1)
import pdb
pdb.set_trace()
# get numerica values
data_ukbio = df_UKBIO.values
# Get the predictions on the BIOBANK dataset
results_biobank = exported_pipeline.predict(testing_features)
