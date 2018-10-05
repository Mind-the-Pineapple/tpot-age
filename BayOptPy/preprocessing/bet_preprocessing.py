#!/usr/bin/env python

import os
import re

import numpy as np
import nibabel as nib
from nipype.interfaces.fsl import BET
from nipype import SelectFiles, Node
from nipype.pipeline.engine import Workflow
from nipype.interfaces.io import DataSink, DataGrabber


def get_mean_image(dataPath, rawsubjectsId):
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
    nib.save(niiMeanImg, os.path.join(dataPath, 'mean_img.nii'))

def bet_nipype_pipeline(dataPath):
    # Get image
    #-----------------------------------------------------------------------------
    dataSource = Node(interface=DataGrabber(outfields=['t1']),
                                            name='DataSource')
    dataSource.inputs.base_directory = os.getcwd()
    dataSource.inputs.template = 'data/mean_img.nii'
    dataSource.inputs.sort_filelist = True

    bet = Node(BET(), name='bet')
    bet.inputs.mask = True
    bet.inputs.frac = 0.3
    bet.inputs.vertical_gradient = -1
    bet.inputs.padding = True



    # Workflow
    # -----------------------------------------------------------------------------
    dataSink = Node(DataSink(), name='DataSink')
    dataSink.inputs.base_directory = dataPath
    substitutions = [('_subject_id_', '')]
    dataSink.inputs.substitutions = substitutions

    # Define workflow name and where output will be saved
    preproc = Workflow(name='preprocessed_data')
    preproc.base_dir = dataPath

    # Define connection between nodes
    preproc.connect([
                     (dataSource, bet,         [('t1'        , 'in_file'   )] ),
                     (bet       , dataSink,    [('mask_file' , 'betout.mask'  )] ),
                     (bet       , dataSink,    [('out_file'  , 'betout.output')] )
                   ])

    preproc.run()


if __name__ == '__main__':
    cwd = os.getcwd()
    print('Current working directory is: %s' %cwd)

    # Get nii images and list of subjects
    dataPath = os.path.join(cwd, 'data')
    # remove the file end and get list of all used subjects
    fileList = os.listdir(dataPath)
    rawsubjectsId = [re.sub(r'^smwc1(.*?)\_mpr-1_anon.nii$', '\\1', file) for file in fileList if file.endswith('.nii')]
    # TODO: Change this. For testing purpose select just the first 5 subjects
    rawsubjectsId = rawsubjectsId[:5]

    # get main image from the dataset
    get_mean_image(dataPath, rawsubjectsId)

    # get brain mask for the mean image
    bet_nipype_pipeline(dataPath)

