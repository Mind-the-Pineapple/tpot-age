# General function to compare the BANC and UKBIO BANC input

import pandas as pd
import numpy as np
import pdb

from BayOptPy.helperfunctions import get_paths, get_data


# Load both free surfer files
debug = False
resamplefactor = 1
project_ukbio_wd, project_data_ukbio, _ = get_paths(debug, 'UKBIO_freesurf')
project_banc_wd, project_data_banc, _  = get_paths(debug, 'BANC_freesurf')
demographics_banc, imgs_banc, data_banc, freesurfer_df_banc = get_data(project_data_banc, 'BANC_freesurf', debug,
                                                                       project_banc_wd, resamplefactor)
demographics_ukbio, imgs_ukbio, data_ukbio, freesurfer_df_ukbio = get_data(project_data_ukbio, 'UKBIO_freesurf', debug,
                                                                           project_ukbio_wd, resamplefactor)

# check the columsn between both datasets
# First print the size of dataset
print(freesurfer_df_banc.shape, freesurfer_df_ukbio.shape)

# Save the colunns as varaible
freesurfer_banc_columns = list(freesurfer_df_banc)
freesurfer_ukbio_columns = list(freesurfer_df_ukbio)

# Check the banc dataset consist of a reduced set from the biobanc dataset
print('Columns that are missing in the ukbiobanc but are on the banc dataset')
print(set(freesurfer_banc_columns).difference(freesurfer_ukbio_columns))
print('Columns that are missing in the bancdataset but are on the ukbiobanc')
print(set(freesurfer_ukbio_columns).difference(freesurfer_banc_columns))
print('Columns that are ')

print('I am done!')
# Compare the columns entry
