import os
import re
import pandas as pd
from nilearn import masking, image
import nibabel as nib
import numpy as np
from tqdm import tqdm

def get_paths(debug, dataset):

    if debug and dataset == 'OASIS':
        project_wd = os.getcwd()
        project_data = os.path.join(project_wd, 'data')
    elif debug and dataset == 'BANC':
        project_wd = os.getcwd()
        project_data = os.path.join(os.getenv('HOME'), 'NaN')
    elif not debug:
        project_wd = '/code'
        project_data = os.path.join(os.sep, 'data')
    else:
        raise ValueError('Analysis for this dataset is not yet implemented!')

    project_sink = os.path.join(project_data, 'output')
    print('Code Path: %s' %project_wd)
    print('Data Path: %s' %project_data)
    print('Data Out: %s' %project_sink )
    return project_wd, project_data, project_sink


def get_data_covariates(dataPath, rawsubjectsId, dataset):
    if dataset == 'OASIS':
        # Load the demographic details from the dataset
        demographics = pd.read_csv(os.path.join(dataPath, 'oasis_cross-sectional.csv'))
        # sort demographics by ascending id
        demographics = demographics.sort_values('ID')

        # Check if there is any subject for which we have the fmri data but no demographics
        missingsubjectsId = list(set(demographics['ID']) ^ set(rawsubjectsId))
        # remove the demographic data from the missing subjects
        demographics = demographics.loc[~demographics['ID'].isin(missingsubjectsId)]

        # list of subjects that do not have dementia (CDR > 0)
        selectedSubId = demographics.loc[(demographics['CDR'] == 0) | (demographics['CDR'].isnull()), 'ID']
        # filter demographics to exclude those with CDR > 0
        demographics = demographics.loc[demographics['ID'].isin(selectedSubId)]

    elif dataset == 'BANC':
        # Load the demographic details from the dataset
        column_names = ['ID', 'original_dataset', 'sex', 'Age']
        demographics = pd.read_csv(os.path.join(dataPath, 'BANC_2016.csv'), names=column_names)

        # Check if there is any subject for which we have the fmri data but no demographics
        missingsubjectsId = list(set(demographics['ID']) ^ set(rawsubjectsId))
        # remove the demographic data from the missing subjects
        demographics = demographics.loc[~demographics['ID'].isin(missingsubjectsId)]
        selectedSubId = list(set(demographics['ID']) ^ set(missingsubjectsId))

    else:
        raise ValueError('Analysis for this dataset is not yet implemented!')

    return demographics, selectedSubId


def get_data(project_data, dataset):
    print('Getting data')
    if dataset == 'OASIS':
        # remove the file end and get list of all used subjects
        fileList = os.listdir(project_data)
        rawsubjectsId = [re.sub(r'^smwc1(.*?)\_mpr-1_anon.nii$', '\\1', file) for file in fileList if file.endswith('.nii')]
        # TODO: Change this. For testing purpose select just the first 5 subjects
        #rawsubjectsId = rawsubjectsId[:25]

        # Load the demographics for each subject
        demographics, selectedSubId = get_data_covariates(project_data, rawsubjectsId, dataset)

        # Load image proxies
        imgs = [nib.load(os.path.join(project_data, 'smwc1%s_mpr-1_anon.nii' %subject)) for subject in tqdm(selectedSubId)]

        # resample dataset to a lower quality. Increase the voxel size by two
        resampleby2affine = np.array([[2, 1, 1, 1], [1, 2, 1, 1], [1, 1, 2, 1], [1, 1, 1, 1]])
        resampledimgs = [image.resample_img(img, np.multiply(img.affine, resampleby2affine)) for img in imgs]

        # Use nilearn to mask only the brain voxels across subjects
        print('Compute brain mask')
        MeanImgMask = masking.compute_multi_epi_mask(resampledimgs, lower_cutoff=0.001, upper_cutoff=.7, opening=1)
        # Apply the group mask on all subjects.
        # Note: The apply_mask function returns the flattened data as a numpy array
        maskedData = [masking.apply_mask(img, MeanImgMask) for img in resampledimgs]
        print('Applied mask to the dataset')

        # Transform the imaging data into a np array (subjects x voxels)
        maskedData = np.array(maskedData)
    elif dataset == 'BANC':
        project_data_path = os.path.join(project_data, 'wm_data')
        # remove the file end and get list of all used subjects
        fileList = os.listdir(project_data_path)
        rawsubjectsId = [file[5:12] for file in fileList if file.endswith('.nii.gz')]
        # TODO: Change this. For testing purpose select just the first 5 subjects
        #rawsubjectsId = rawsubjectsId[:25]

        # Load the demographics for each subject
        demographics, selectedSubId = get_data_covariates(project_data, rawsubjectsId, dataset)

        # Load image proxies
        imgs = [nib.load(os.path.join(project_data_path, file)) for file in tqdm(fileList) if file[5:12] in selectedSubId]

        resamplefactor = 1
        print('Resample the dataset by a factor of %d' %resamplefactor)
        # resample dataset to a lower quality. Increase the voxel size by two
        resampleby2affine = np.array([[resamplefactor, 1, 1, 1],
                                      [1, resamplefactor, 1, 1],
                                      [1, 1, resamplefactor, 1],
                                      [1, 1, 1, 1]])
        resampledimgs = [image.resample_img(img, np.multiply(img.affine, resampleby2affine)) for img in imgs]

        # Use nilearn to mask only the brain voxels across subjects
        print('Compute brain mask')
        #The lower and the upper_cutoff represent the lower and the upper fraction of the histogram to be discarded
        MeanImgMask = masking.compute_multi_epi_mask(resampledimgs, lower_cutoff=0.001, upper_cutoff=.85, opening=False)
        # Apply the group mask on all subjects.
        # Note: The apply_mask function returns the flattened data as a numpy array
        maskedData = [masking.apply_mask(img, MeanImgMask) for img in resampledimgs]
        # to save an nifti image of the image
        # nib.
        print('Applied mask to the dataset')

        # Transform the imaging data into a np array (subjects x voxels)
        maskedData = np.array(maskedData)

    else:
        raise ValueError('Analysis for this dataset is not yet implemented!')
    return demographics, imgs, maskedData