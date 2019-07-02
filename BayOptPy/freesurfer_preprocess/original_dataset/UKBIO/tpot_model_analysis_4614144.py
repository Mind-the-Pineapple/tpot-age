import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import RobustScaler
from tpot.builtins import StackingEstimator

from BayOptPy.helperfunctions import (get_paths, get_data,
                                      drop_missing_features,
                                      set_publication_style)

"""
This script tests the best model recommened by the combined dataset (UKBIO +
BANC) for 100 generations, random
seed 20, initial population 1000, mutation rate and cross-validation rate 0.9
and cross-over 0.1
"""

# General Settings
#-------------------------------------------------------------------------------
debug = False
resamplefactor = 1
random_seed = 20
save_path = '/code/BayOptPy/tpot/Output/random_seed/100_generations/random_seed_%03d/' %(random_seed)
# Load the combined dataset
project_wd, project_data, _ = get_paths(debug, 'freesurf_combined')
demographics, _, df_data =  \
             get_data(project_data, 'freesurf_combined', debug,
                      project_wd, resamplefactor, raw=False, analysis=None)
# Drop the last column that corresponds the name of the dataset
df_data = df_data.drop('dataset', axis=1)
#-------------------------------------------------------------------------------
# Train the model with BANC
#-------------------------------------------------------------------------------
targetAttribute = demographics[['age']]
demographics = demographics.set_index('id')

# Add a few of the BIOBANK Dataset into the training set
Xtrain, Xtemp, Ytrain, Ytemp = train_test_split(df_data, targetAttribute,
                                                test_size=.90,
                                                stratify=demographics['stratify'],
                                                random_state=random_seed)
train_demographics = demographics.loc[Xtemp.index]
Xvalidate, Xtest, Yvalidate, Ytest = train_test_split(Xtemp, Ytemp,
                                                test_size=.05,
                                                stratify=train_demographics['stratify'],
                                                random_state=random_seed)


print('Divided BANC dataset into test and training')
print('Check train test split sizes')
print('X_train: ' + str(Xtrain.shape))
print('X_test: '  + str(Xtest.shape))
print('Y_train: ' + str(Ytrain.shape))
print('Y_test: '  + str(Ytest.shape))
print('X_valitation ' + str(Xvalidate.shape))
print('Y_test: ' + str(Yvalidate.shape))

# Normalise the test dataset and apply the transformation to the train
# dataset
robustscaler = RobustScaler().fit(Xtrain)
Xtrain_scaled = robustscaler.transform(Xtrain)
Xtest_scaled = robustscaler.transform(Xtest)
Xvalidate_scaled = robustscaler.transform(Xvalidate)

# Transform pandas into numpy arrays (no nneed to do it if you are scaling
# the results)
Ytrain = Ytrain.values
Ytest = Ytest.values
Yvalidate = Yvalidate.values

# Best pipeline recommended by TPOT
exported_pipeline = make_pipeline(
      StackingEstimator(estimator=LinearRegression()),
      StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=True,
                                                      max_features=0.9000000000000001,
                                                      min_samples_leaf=3,
                                                      min_samples_split=10,
                                                      n_estimators=100,
                                                      random_state=42)),
      ExtraTreesRegressor(bootstrap=False,
                          max_features=0.55,
                          min_samples_leaf=5,
                          min_samples_split=17,
                          n_estimators=100,
                          random_state=42)
                                 )

exported_pipeline.fit(Xtrain_scaled, Ytrain)
print('Print MAE - test')
y_predicted = exported_pipeline.predict(Xtest_scaled)
mae = mean_absolute_error(Ytest, y_predicted)
print(mae)
print('Print MAE - training')
y_predicted_train = exported_pipeline.predict(Xtrain_scaled)
mae_train = mean_absolute_error(Ytrain, y_predicted_train)
print(mae_train)
print('Print MAE - validation')
y_predicted_validation = exported_pipeline.predict(Xvalidate_scaled)
mae_validation = mean_absolute_error(Yvalidate, y_predicted_validation)
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
plt.scatter(Yvalidate, y_predicted_validation)
plt.ylabel('Predicted Age')
plt.xlabel('True Age')
plt.savefig(os.path.join(save_path, 'validation_predicted_true_age.png'))
plt.close()
