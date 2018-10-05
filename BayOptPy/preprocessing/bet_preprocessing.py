#!/usr/bin/env python

import os
import re
from nipype.interfaces.fsl import BET
from nipype import SelectFiles, Node
from nipype.pipeline.engine import Workflow
from nipype.interfaces.io import DataSink, DataGrabber
from nipype.interfaces.utility import IdentityInterface

# Data location
print('current working directory is: %s' %os.getcwd())
dataPath = 'data'

# Get image
#-----------------------------------------------------------------------------
# get list of subjects, select all files with .nii ending
fileList = os.listdir(dataPath)
subjects_id = [re.sub(r'^smwc1(.*?)\_mpr-1_anon.nii$', '\\1', file) for file in
            fileList if file.endswith('.nii')]
# TODO: select just the first 5 subjects
subjects_id = subjects_id[:1]
dataSource = Node(interface=DataGrabber(infields=['subject_id'],
                                        outfields=['t1']),
                  name='DataSource')
dataSource.inputs.base_directory = os.getcwd()
dataSource.inputs.template = 'data/smwc1%s_mpr-1_anon.nii'
dataSource.inputs.sort_filelist = True
# this specifies the variables for the field_templates. Or in other words, what
# to iterate over
dataSource.inputs.subject_id = subjects_id

bet = Node(BET(), name='bet')
bet.inputs.mask = True

# Workflow
# -----------------------------------------------------------------------------
# Define Infosource, which is the input Node. Information from subject_id are
# obtained from this Node
infoSource = Node(interface=IdentityInterface(fields=['subject_id']),
             name='InfoSource')
infoSource.iterables = ('subject_id', subjects_id)

dataSink = Node(DataSink(), name='DataSink')
dataSink.inputs.base_directory = dataPath
substitutions = [('_subject_id_', '')]
dataSink.inputs.substitutions = substitutions

# Define workflow name and where output will be saved
preproc = Workflow(name='preprocessed_data')
preproc.base_dir =  dataPath

# Define connection between nodes
preproc.connect([
                 (infoSource, dataSource, [('subject_id', 'subject_id')] ),
                 (dataSource, bet,        [('t1'        , 'in_file'   )] ),
                 (bet       , dataSink,    [('mask_file' , 'betout.mask'  )] ),
                 (bet       , dataSink,    [('out_file'  , 'betout.output')] )
               ])

preproc.run()


