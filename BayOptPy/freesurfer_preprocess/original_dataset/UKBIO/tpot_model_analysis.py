import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.metrics import mean_absolute_error
from tpot.builtins import StackingEstimator

from BayOptPy.helperfunctions import (get_paths, get_data,
                                      drop_missing_features,
                                      set_publication_style)

"""
This script tests the best model recommened by TPOT for 100 generations, random
seed 20, initial population 1000, mutation rate and cross-validation ratexxx
"""

# General Settings
#-------------------------------------------------------------------------------
debug = False
resamplefactor = 1
random_seed = 20
save_path = '/code/BayOptPy/'
# Load the clean data for both the UKBIO and the BANC analysis
# This version of the UKBIOBANK dataset contains the same columns as the BANC
# dataset
project_ukbio_wd, project_data_ukbio, _ = get_paths(debug, 'UKBIO_freesurf')
_, _, df_ukbio =  \
             get_data(project_data_ukbio, 'UKBIO_freesurf', debug,
                      project_ukbio_wd, resamplefactor, raw=False)
df_ukbio = df_ukbio.set_index('ID')
# Drop the last column that corresponds the name of the dataset
df_ukbio = df_ukbio.drop('dataset', axis=1)

project_banc_wd, project_banc_data, _ = get_paths(debug,'BANC_freesurf')
demographics_banc,__, df_banc = get_data(project_banc_data,
                                                'BANC_freesurf',
                                                debug, project_banc_wd,
                                                resamplefactor, raw=False)
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
print('Print BANC MAE')
print(mae_banc)

# plot predicted vs true for the BIOBANK
fig = plt.figure()
plt.scatter(Ytest, y_predicted_banc)
plt.ylabel('Predicted Age')
plt.xlabel('True Age')
plt.savefig(os.path.join(save_path, 'banc_predicted_true_age.png'))
plt.close()

#-------------------------------------------------------------------------------
# Test the trained model on the BIOBANK
#-------------------------------------------------------------------------------
data_ukbio = df_ukbio.values
# Get the predictions on the BIOBANK dataset
y_predicted_biobank = exported_pipeline.predict(data_ukbio)
mae_biobank = mean_absolute_error(targetAttribute_ukbio, y_predicted_biobank)
print('Print UKBIO MAE')
print(mae_biobank)

# plot predicted vs true for the BIOBANK
fig = plt.figure()
plt.scatter(targetAttribute_ukbio, y_predicted_biobank)
plt.ylabel('Predicted Age')
plt.xlabel('True Age')
plt.savefig(os.path.join(save_path, 'biobank_predicted_true_age.png'))
plt.close()
