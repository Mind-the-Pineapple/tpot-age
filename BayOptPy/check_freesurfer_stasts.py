import pandas as pd
import os
import numpy as np

from BayOptPy.helperfunctions import get_data_covariates, get_paths

# Load the freesurfer aparc stats, remove the file endings and analyse the number of subjects
print('Analysing and cleaning the aparc - LH file')
aparc_data_lh = pd.read_csv('data/freesurfer_BANC2016/aparc_stats_lh.txt', sep='\t')
aparc_data_lh['lh.aparc.thickness'] = aparc_data_lh['lh.aparc.thickness'].str.replace(r'_*.nii', '')
# get rid of the T1
aparc_data_lh['lh.aparc.thickness'] = aparc_data_lh['lh.aparc.thickness'].str.replace(r'_*.T1', '')
# split the BANC id from its original dataset ID
aparc_data_lh['BANC_ID'] = aparc_data_lh['lh.aparc.thickness'].str.split('_').str[0]
aparc_data_lh['orig_dataset_ID'] = aparc_data_lh['lh.aparc.thickness'].str.split('_').str[1]
print('aparc shape: (%d, %d)' %(aparc_data_lh.shape[0], aparc_data_lh.shape[1]))
print(' ')

print('Analysing and cleaning the aparc - RH file')
aparc_data_rh = pd.read_csv('data/freesurfer_BANC2016/aparc_stats_rh.txt', sep='\t')
aparc_data_rh['rh.aparc.area'] = aparc_data_rh['rh.aparc.area'].str.replace(r'_*.nii', '')
# get rid of the T1
aparc_data_rh['rh.aparc.area'] = aparc_data_rh['rh.aparc.area'].str.replace(r'_*.T1', '')
# split the BANC id from its original dataset ID
aparc_data_rh['BANC_ID'] = aparc_data_rh['rh.aparc.area'].str.split('_').str[0]
aparc_data_rh['orig_dataset_ID'] = aparc_data_rh['rh.aparc.area'].str.split('_').str[1]
print('aparc shape: (%d, %d)' %(aparc_data_rh.shape[0], aparc_data_rh.shape[1]))

# Check if the LH and the RH have the same subjects
if aparc_data_rh.shape[0] == aparc_data_lh.shape[0]:
    print('Both hemispheres have the same number of subjects')
else:
    raise ValueError('The LH and RH hemisphere have different number of subjects')
if np.sum(aparc_data_rh['rh.aparc.area'] == aparc_data_lh['lh.aparc.thickness']) == aparc_data_rh.shape[0]:
    print('Both LH and RH are analysing the same subjects')
print(' ')


# Load the freesurfer aseg stats and do the same analysis done for the aparc file
print('Analysing and cleaning the aseg file')
aseg_data = pd.read_csv('data/freesurfer_BANC2016/aseg_stats.txt', sep='\t')
# Process it in the same way as the aparc dataset
aseg_data['Measure:volume'] = aseg_data['Measure:volume'].str.replace(r'_*.nii', '')
# get rid of the T1
aseg_data['Measure:volume'] = aseg_data['Measure:volume'].str.replace(r'_*.T1', '')
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
missa = np.sort(list(set(selectedSubId) - set(aparc_data_lh['BANC_ID'].values)))
missb = np.sort(list(set(aparc_data_lh['BANC_ID'].values) - set(selectedSubId)))
print('Number of missing subjects in the BANC dataset: %d' %len(missa))
print('Number of missing subjects in the freesurfer dataset: %d' %len(missb))
# number of subjects from the CanCan list
print('List of subjects on the cluster that were not analysed')
print(missa)

