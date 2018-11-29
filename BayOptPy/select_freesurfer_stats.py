import pandas as pd
import os
import numpy as np

from BayOptPy.helperfunctions import get_data_covariates, get_paths


# This script assumes that you already have the aseg and aparc lh and rh for the subjects of interest. It then selects
# the stats for the subjects that are present in the BANC2016 dataset

# Load the freesurfer aparc stats, remove the file endings and analyse the number of subjects
print('Analysing and cleaning the aparc - LH file')
aparc_data_lh = pd.read_csv('data/freesurfer_BANC2016/aparc_stats_lh.txt', sep='\t')
aparc_data_lh['lh.aparc.thickness'] = aparc_data_lh['lh.aparc.thickness'].str.replace(r'_*.nii', '')
# get rid of the T1
aparc_data_lh['lh.aparc.thickness'] = aparc_data_lh['lh.aparc.thickness'].str.replace(r'_*.T1', '')
# split the BANC id from its original dataset ID
aparc_data_lh['BANC_ID'] = aparc_data_lh['lh.aparc.thickness'].str.split('_').str[0]
aparc_data_lh['orig_dataset_ID'] = aparc_data_lh['lh.aparc.thickness'].str.split('_').str[1]
# use the BANC_ID as pandas index
aparc_data_lh = aparc_data_lh.set_index('BANC_ID')
print('aparc shape: (%d, %d)' %(aparc_data_lh.shape[0], aparc_data_lh.shape[1]))
print(' ')

print('Analysing and cleaning the aparc - RH file')
aparc_data_rh = pd.read_csv('data/freesurfer_BANC2016/aparc_stats_rh.txt', sep='\t')
aparc_data_rh['rh.aparc.thickness'] = aparc_data_rh['rh.aparc.thickness'].str.replace(r'_*.nii', '')
# get rid of the T1
aparc_data_rh['rh.aparc.thickness'] = aparc_data_rh['rh.aparc.thickness'].str.replace(r'_*.T1', '')
# split the BANC id from its original dataset ID
aparc_data_rh['BANC_ID'] = aparc_data_rh['rh.aparc.thickness'].str.split('_').str[0]
# use the BANC_ID as pandas index
aparc_data_rh = aparc_data_rh.set_index('BANC_ID')
print('aparc shape: (%d, %d)' %(aparc_data_rh.shape[0], aparc_data_rh.shape[1]))

# Check if the LH and the RH have the same subjects
if aparc_data_rh.shape[0] == aparc_data_lh.shape[0]:
    print('Both hemispheres have the same number of subjects')
else:
    raise ValueError('The LH and RH hemisphere have different number of subjects')
if np.sum(aparc_data_rh['rh.aparc.thickness'] == aparc_data_lh['lh.aparc.thickness']) == aparc_data_rh.shape[0]:
    print('Both LH and RH are analysing the same subjects')
print(' ')


# Load the freesurfer aseg stats and do the same analysis done for the aparc file
print('Analysing and cleaning the aseg file')
aseg_data = pd.read_csv('data/freesurfer_BANC2016/aseg_stats.txt', sep='\t')
# Process it in the same way as the aparc dataset
aseg_data['Measure:volume'] = aseg_data['Measure:volume'].str.replace(r'_*.nii', '')
# get rid of the T1
aseg_data['Measure:volume'] = aseg_data['Measure:volume'].str.replace(r'_*.T1', '')
# split the BANC id from its original dataset ID
aseg_data['BANC_ID'] = aseg_data['Measure:volume'].str.split('_').str[0]
# use the BANC_ID as pandas index
aseg_data = aseg_data.set_index('BANC_ID')
print('aseg shape: (%d, %d)' %(aseg_data.shape[0], aseg_data.shape[1]))

print('List of subjects present on the aparc but not on the aseg stats')
print(list(set(aparc_data_lh['lh.aparc.thickness']) - set(aseg_data['Measure:volume'])))
print(' ')

# get the list of subjects and demographics for the BANC dataset
debug = True
dataset = 'BANC'
project_wd, project_data, project_sink = get_paths(debug, dataset)
project_data_path = os.path.join(project_data, 'wm_data')
# remove the file end and get list of all used subjects
fileList = os.listdir(project_data_path)
rawsubjectsId = [file[5:12] for file in fileList if file.endswith('.nii.gz')]
demographics, selectedSubId = get_data_covariates(project_data, rawsubjectsId, dataset)

# Compare the list of subjects from the BANC and the free-surfer analyis
missa = np.sort(list(set(selectedSubId) - set(aparc_data_lh.index.values)))
missb = np.sort(list(set(aparc_data_lh.index.values) - set(selectedSubId)))
print('Number of missing subjects in the BANC dataset: %d' %len(missa))
print('Number of missing subjects in the freesurfer dataset: %d' %len(missb))

# Get the aparc and aseg stasts for the subjects in the BANC2016 dataset and filter them for those subjects
# Note: As I have already checked that the LH and RH hemisphere consist of an analysis of the same subjects, I am using
# the subjects from the LH to perform the selection
common_subjects = list(set(selectedSubId).intersection(aparc_data_lh.index.values))
aparc_data_lh_BANC2016 = aparc_data_lh[aparc_data_lh.index.isin(common_subjects)]
aparc_data_rh_BANC2016 = aparc_data_rh[aparc_data_rh.index.isin(common_subjects)]
aseg_data_BANC2016 = aseg_data[aseg_data.index.isin(common_subjects)]
# check if both aparcs have the same size
assert(aparc_data_lh_BANC2016.shape == aparc_data_rh_BANC2016.shape)
# check if the same subjects are present in the aseg and aparc dataset
assert(np.sum(aparc_data_lh_BANC2016.index == aseg_data_BANC2016.index) == aparc_data_lh_BANC2016.shape[0])

# For each subject combine both aparc and aseg stats
freesurfer_stats = pd.concat([aparc_data_lh_BANC2016, aparc_data_rh_BANC2016, aseg_data_BANC2016], axis=1, sort=False)
# select only the numeric values
freesurfer_stats_num = freesurfer_stats._get_numeric_data()
# dump the results as csv
freesurfer_stats_num.to_csv('aparc_aseg_stats_BANC.csv')
print('Done')
