import os
import re
import pandas as pd
from multiprocessing import Process, Pool
from functools import partial
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
    elif not debug and dataset == 'OASIS':
        project_wd = '/code'
        project_data = os.path.join(os.sep, 'data')
    elif not debug and dataset == 'BANC':
        project_wd = '/code'
        project_data = os.path.join(os.sep, 'data', 'BANC_2016')
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
        selectedSubId = rawsubjectsId

    else:
        raise ValueError('Analysis for this dataset is not yet implemented!')

    return demographics, selectedSubId

def _multiprocessing_resample(img, target_affine):
    resampled_img = image.resample_img(img, target_affine=target_affine,
                                       interpolation='nearest')
    return resampled_img

def _load_nibabel(filePath):
    img = nib.load(os.path.join(filePath))
    return img

def get_data(project_data, dataset, debug, project_wd, resamplefactor):
    print('Loading Brain image data')
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

    elif dataset == 'BANC':
        # For now, performing analysis on White Matter.
        project_data_path = os.path.join(project_data, 'wm_data')
        # remove the file end and get list of all used subjects
        fileList = os.listdir(project_data_path)
        rawsubjectsId = [file[5:12] for file in fileList if file.endswith('.nii.gz')]
        # TODO: select only a set of 5 subjects
        # rawsubjectsId = rawsubjectsId[:5]

        # Load the demographics for each subject
        demographics, selectedSubId = get_data_covariates(project_data, rawsubjectsId, dataset)
        # Get the file path of the selected subjects
        subjectsFile = [os.path.join(project_data_path, file) for file in fileList if file[5:12] in selectedSubId]

        # Load image proxies
        with Pool() as p:
            imgs = list(tqdm(p.imap(_load_nibabel, subjectsFile), total=len(selectedSubId)))
    else:
        raise ValueError('Analysis for this dataset is not yet implemented!')

    print('Resample the dataset by a factor of %d' %resamplefactor)
    print('Original image size: %s' %(imgs[0].shape,))
    # resample dataset to a lower quality. Increase the voxel size by two
    resampleby2affine = np.array([[resamplefactor, 1, 1, 1],
                                  [1, resamplefactor, 1, 1],
                                  [1, 1, resamplefactor, 1],
                                  [1, 1, 1, 1]])
    target_affine = np.multiply(imgs[0].affine, resampleby2affine)
    print('Resampling Images')
    with Pool() as p:
        args = partial(_multiprocessing_resample, target_affine=target_affine)
        resampledimgs = list(tqdm(p.imap(args, imgs), total=len(imgs)))
    print('Resampled image size: %s' %(resampledimgs[0].shape,))

    # Use nilearn to mask only the brain voxels across subjects
    print('Compute brain mask')
    #The lower and the upper_cutoff represent the lower and the upper fraction of the histogram to be discarded
    MeanImgMask = masking.compute_multi_epi_mask(resampledimgs, lower_cutoff=0.001, upper_cutoff=.85, opening=False)
    # Apply the group mask on all subjects.
    # Note: The apply_mask function returns the flattened data as a numpy array
    maskedData = [masking.apply_mask(img, MeanImgMask) for img in resampledimgs]
    # If debug option is set, save an nifti image of the image.
    # Note: if you resampled the image you will not be able to overlay it on the original brain
    if debug:
        mask_path = os.path.join(project_wd, 'BayOptPy', 'tpot')
        print('Saving brain mask: %s' %mask_path)
        nib.save(MeanImgMask, os.path.join(mask_path, 'mask_%s.nii.gz' %dataset))
    print('Applied mask to the dataset')

    # Transform the imaging data into a np array (subjects x voxels)
    maskedData = np.array(maskedData)

    return demographics, imgs, maskedData