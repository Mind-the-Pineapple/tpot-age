#!/usr/bin/env python

import os
import re
import pandas as pd
from nilearn import masking
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
import nibabel as nib


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


if __name__ == '__main__':

    cwd = os.getcwd()
    print('Current working directory is: %s' %cwd)

    # Get nii images and list of subjects
    dataPath = os.path.join(cwd, 'data')
    # remove the file end and get list of all used subjects
    fileList = os.listdir(dataPath)
    rawsubjectsId = [re.sub(r'^smwc1(.*?)\_mpr-1_anon.nii$', '\\1', file) for file in fileList if file.endswith('.nii')]
    # TODO: Change this. For testing purpose select just the first 5 subjects
    rawsubjectsId = rawsubjectsId[:250]

    # Load the demographics for each subject
    demographics = get_data_covariates(dataPath, rawsubjectsId)

    # Load image proxies
    imgs = [nib.load(os.path.join(dataPath, 'smwc1%s_mpr-1_anon.nii' %subject)) for subject in rawsubjectsId]

    # Use nilearn to mask only the brain voxels across subjects
    MeanImgMask = masking.compute_multi_epi_mask(imgs, lower_cutoff=0.001, upper_cutoff=.7, opening=1)
    # Apply the group mask on all subjects.
    # Note: The apply_mask function returns the flattened data as a numpy array
    maskedData = [masking.apply_mask(img, MeanImgMask) for img in imgs]
    print('Applied mask to the dataset')
    # Transform the imaging data into a dataframe (subjects x voxels)
    imgDf = pd.DataFrame(data=maskedData)

    # split train-test dataset
    targetAttribute = demographics['Age']
    Xtrain, Xtest, Ytrain, Ytest = model_selection.train_test_split(imgDf, targetAttribute, test_size=.4, random_state=42)
    print('Divided dataset into test and training')
    print('Check train test split sizes')
    print('X_train: ' + str(Xtrain.shape))
    print('X_test: '  + str(Xtest.shape))
    print('Y_train: ' + str(Ytrain.shape))
    print('Y_test: '  + str(Ytest.shape))

    # Do simple Linear regression
    linReg = LinearRegression()
    linReg.fit(Xtrain, Ytrain)
    print('Performed simple linear regresssion')
    Ypred = linReg.predict(Xtest)
    # plot the results
    plt.scatter(Ytest, Ypred)
    plt.xlabel("Age: $Y_i$")
    plt.ylabel("Predicted Age: $\hat{Y}_i$")
    plt.title("Age vs Predicted Age: $Y_i$ vs $\hat{Y}_i$")
    plt.show()
    print('Done')