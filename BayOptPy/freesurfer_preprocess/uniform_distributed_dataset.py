# This script assumes taht the freesurfer csv for the BANC data has already been generated
import os
import pandas as pd
import numpy as np
import pdb
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

from BayOptPy.helperfunctions import get_paths, get_data, drop_missing_features

def str_to_bool(s):
    '''
    As arg pass does not acess boolen, transfrom the string into
    booleans
    '''
    if s == 'True':
        return True
    elif s == 'False':
        return False
#-----------------------------------------------------------------------------
# Settings
#-----------------------------------------------------------------------------
debug = False
dataset = 'freesurf_combined'
resamplefactor = 1
save_path = os.path.join('/code/BayOptPy', 'freesurfer_preprocess')
raw = 'False'
analysis = 'uniform'
project_wd, project_data, project_sink = get_paths(debug, dataset)

demographics, imgs, dataframe  = get_data(project_data, dataset,
                                        debug, project_wd,
                                        resamplefactor,
                                        raw=str_to_bool(raw),
                                        analysis=analysis)

# transform age into ints
demographics['age_int'] = demographics['age'].astype('int32', copy=False)

# Select 14 subjects for all ages that have 14 representatives.
age_range = np.arange(demographics['age'].min(), demographics['age'].max())
# remove entry where you don't have 14 subjects
max_n = 14
age_to_remove = [35, 36, 39, 42, 78, 79, 80, 81, 82, 83, 85, 89]
age_range = np.setdiff1d(age_range, age_to_remove)
# iterate over the dataframe and select 14 subjects for each age range
ids_to_use = []
for age in age_range:
    ids_to_use.append(demographics.index[demographics['age_int'] ==
                                         age].tolist()[:max_n])

# flatten ids_to_use
ids_to_use = [item for sublist in ids_to_use for item in sublist]
# Filter the demographics dataframe
demographics = demographics[demographics.index.isin(ids_to_use)]
# set subject's id as index
demographics = demographics.set_index('id')
# filter dataset using index of the subjects
dataframe = dataframe.loc[demographics.index]

# Print some diagnosis
print('Shape of the new demographics:')
print(demographics.shape)
print('Oldest %d and youngest %d subject' %(demographics['age_int'].max(),
                                            demographics['age_int'].min()))
print('Number of age bins %d' %len(demographics['age_int'].unique()))
import pdb
pdb.set_trace()
print('Done')
