# General function to compare the BANC and UKBIO BANC input

import pandas as pd
import numpy as np

from BayOptPy.helperfunctions import get_paths, get_data


# Load both free surfer files
debug = True
resamplefactor = 1
project_ukbio_wd, project_data_ukbio, _ = get_paths(debug, 'UKBIO_freesurf')
project_banc_wd, project_data_banc, _  = get_paths(debug, 'BANC_freesurf')
demographics_banc, imgs_banc, data_banc, freesurfer_df_banc = get_data(project_data_banc, 'BANC_freesurf', debug,
                                                                       project_banc_wd, resamplefactor)
demographics_ukbio, imgs_ukbio, data_ukbio, freesurfer_df_ukbio = get_data(project_data_ukbio, 'UKBIO_freesurf', debug,
                                                                           project_ukbio_wd, resamplefactor)

print('I am done!')
# Compare the columns entry
