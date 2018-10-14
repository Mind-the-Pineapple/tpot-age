import os
import re
import pandas as pd
from nilearn import masking, image
import nibabel as nib
import numpy as np

def get_data_covariates(dataPath, rawsubjectsId):
    # Load the demographic details from the dataset
    demographics = pd.read_csv(os.path.join(dataPath, 'oasis_cross-sectional.csv'))
    # sort demographics by ascending id
    demographics = demographics.sort_values('ID')

    # Check if there is any subject for which we have the fmri data but no demographics
    missingsubjectsId = list(set(demographics['ID']) ^ set(rawsubjectsId))
    # remove the demographic data from the missing subjects
    demographics = demographics.loc[~demographics['ID'].isin(missingsubjectsId)]

    return demographics


def get_data(project_wd):
    print('Getting data')
    project_data = os.path.join(project_wd, 'data')
    project_sink = os.path.join(project_wd, 'output')

    # remove the file end and get list of all used subjects
    fileList = os.listdir(project_data)
    rawsubjectsId = [re.sub(r'^smwc1(.*?)\_mpr-1_anon.nii$', '\\1', file) for file in fileList if file.endswith('.nii')]
    # TODO: Change this. For testing purpose select just the first 5 subjects
    #rawsubjectsId = rawsubjectsId[:25]

    # Load the demographics for each subject
    demographics = get_data_covariates(project_data, rawsubjectsId)

    # Load image proxies
    imgs = [nib.load(os.path.join(project_data, 'smwc1%s_mpr-1_anon.nii' %subject)) for subject in rawsubjectsId]

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
    return project_sink, demographics, imgs, maskedData