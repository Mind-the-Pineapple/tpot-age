#!/usr/bin/env python

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
import nibabel as nib


def data_preprocessing():
    cwd = os.getcwd()
    dataPath = os.path.join(os.path.dirname(os.path.dirname(cwd)), 'data')
    # remove the file end and get list of all used subjects
    fileList = os.listdir(dataPath)
    rawsubjectsId = [re.sub(r'^smwc1(.*?)\_mpr-1_anon.nii$', '\\1', file) for file in fileList if file.endswith('.nii')]
    # TODO: Change this. For testing purpose select just the first 100 subjects
    rawsubjectsId = rawsubjectsId[:100]

    # Load image proxies
    imgs = [nib.load(os.path.join(dataPath, 'smwc1%s_mpr-1_anon.nii' %subject)) for subject in rawsubjectsId]
    # Load data as numpy array


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
