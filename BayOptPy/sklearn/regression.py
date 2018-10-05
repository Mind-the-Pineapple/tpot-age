#!/usr/bin/env python

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
import nibabel as nib


def data_preprocessing():
    cwd = os.getcwd()
    dataPath = os.path.join(cwd, 'data')
    # remove the file end and get list of all used subjects
    fileList = os.listdir(dataPath)
    rawsubjectsId = [re.sub(r'^smwc1(.*?)\_mpr-1_anon.nii$', '\\1', file) for file in fileList if file.endswith('.nii')]
    # TODO: Change this. For testing purpose select just the first 5 subjects
    rawsubjectsId = rawsubjectsId[:5]

    # Load image proxies
    imgs = [nib.load(os.path.join(dataPath, 'smwc1%s_mpr-1_anon.nii' %subject)) for subject in rawsubjectsId]
    # Load data as numpy array (might get very slow depending on the number of subjects) and reshape the image so that
    # you have one 1D array with the data.
    # Note: np.flatten uses c style ('row-wise') to flatten the array.
    # For example: myarray = np.arange(18).reshape((2,3,3))
    # array([[[ 0,  1,  2],
    #         [ 3,  4,  5],
    #         [ 6,  7,  8]],
    #        [[ 9, 10, 11],
    #         [12, 13, 14],
    #         [15, 16, 17]]])
    # myarray.flatten()
    # array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
    #        17])

    # Create mean image to generate mask
    imgsData = [subjectImg.get_fdata().flatten() for subjectImg in imgs]
    meanImg = np.mean(imgsData, axis=0)
    # reshape data to the correct shape and save image as nii file
    meanImg = meanImg.reshape(imgs[0].get_fdata().shape)
    niiMeanImg = nib.Nifti1Image(meanImg, imgs[0].affine)
    nib.save(niiMeanImg, os.path.join('data','mean_img.nii'))


    # Load demographic details
    demographics = pd.read_csv(os.path.join(dataPath, 'oasis_cross-sectional.csv'))
    # sort demographics by ascending id
    demographics = demographics.sort_values('ID')

    # Check if there is any subject for which we have the fmri data but no demographics
    missingsubjectsId = list(set(demographics['ID']) ^ set(rawsubjectsId))
    # remove the demographic data from the missing subjects
    demographics = demographics.loc[~demographics['ID'].isin(missingsubjectsId)]

    # split train-test dataset
    targetAttribute = demographics['Age']
    Xtrain, Xtest, Ytrain, Ytest = model_selection.train_test_split(demographics, targetAttribute, test_size=.4, random_state=42)
    print('Check train test split sizes')
    print('X_train: ' + str(Xtrain.shape))
    print('X_test: '  + str(Xtest.shape))
    print('Y_train: ' + str(Ytrain.shape))
    print('Y_test: '  + str(Ytest.shape))
    return Xtrain, Xtest, Ytrain, Ytest, demographics


if __name__ == '__main__':
    # Perform preprocessing
    Xtrain, Xtest, Ytrain, Ytest, demographics = data_preprocessing()

    # Do simple Linear regression
    linReg = LinearRegression()
    linReg.fit(Xtrain, Ytrain)
    Ypred = linReg.predict(Xtest)
    # plot the results
    plt.scatter(Ytest, Ypred)
