import pandas as pd
import numpy as np

from BayOptPy.helperfunctions import (create_age_histogram)

# Load demographic details
df = pd.read_csv('/code/UKBIO/UKB_FS_age_sex.csv')
# plot age
create_age_histogram(df['age'], None, 'UKBIO')

