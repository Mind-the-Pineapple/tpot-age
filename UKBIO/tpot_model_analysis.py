import numpy as np
import pandas as pd
import os
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator

from BayOptPy.helperfunctions import get_paths, get_data

debug = False
resamplefactor = 1
project_ukbio_wd, project_data_ukbio, _ = get_paths(debug, 'UKBIO_freesurf')
# demographics_ukbio, imgs_ukbio, data_ukbio, freesurfer_df_ukbio =  \
#             get_data(project_data_ukbio, 'UKBIO_freesurf', debug, project_ukbio_wd, resamplefactor)

df_UKBIO = pd.read_csv(os.path.join(project_data_ukbio, 'UKB_FS_age_sex.csv'))
features = np.array(df_UKBIO['age'])

# This version of the UKBIOBANK dataset contains the same columns as the BANC
# dataset
df_UKBIO = pd.read_csv(os.path.join(project_data_ukbio, 'UKB_10k_FS_4844_adapted.csv'))
# Drop the last column that corresponds the name of the dataset
df_UKBIO = df_UKBIO.drop('dataset', axis=1)
import pdb
pdb.set_trace()
# get numerica values
data_ukbio = df_UKBIO.values

training_features, testing_features, training_target, testing_target = \
            train_test_split(features, data_ukbio, random_state=10)


exported_pipeline = make_pipeline(
    StackingEstimator(estimator=Ridge(alpha=10.0, random_state=42)),
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=True, max_features=0.7500000000000001, min_samples_leaf=4, min_samples_split=4, n_estimators=100, random_state=42)),
    ExtraTreesRegressor(bootstrap=False, max_features=0.9000000000000001, min_samples_leaf=3, min_samples_split=2, n_estimators=100, random_state=42)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
