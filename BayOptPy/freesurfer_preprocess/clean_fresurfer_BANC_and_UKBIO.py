# This script assumes taht the freesurfer csv for the BANC data has already been generated
import os
import pandas as pd
import numpy as np
import pdb
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

from BayOptPy.helperfunctions import get_paths, get_data, drop_missing_features


def visualise_missing_features_banc(freesurfer_df_banc, save_path):
    '''
    Visualise features that are only present in the BANC dataset
    '''
    # just as a proof of concept check that the mean thickness between rh and lh are
    # different
    if sum(freesurfer_df_banc['lh_MeanThickness_thickness'] ==
           freesurfer_df_banc['rh_MeanThickness_thickness']) == len(freesurfer_df_banc):
        print('LH and RH MeanThickness are identical')
    else:
        print('LH and RH MeanThickness are NOT identical')

    # Plot the missing data eTIV1 and eTIV and BrainSegVolNoVent
    # Check if both columns are identical
    data = [freesurfer_df_banc['eTIV'], freesurfer_df_banc['eTIV.1']]
    # If this equality is true it means that all the data is equal
    if sum(freesurfer_df_banc['eTIV']==freesurfer_df_banc['eTIV.1']) == len(freesurfer_df_banc):
        print('eTIV is identical to eTIV.1')
    # plt.figure()
    # plt.boxplot(data)
    # plt.title('eTIV')
    # plt.savefig(os.path.join(save_path, diagnosics, 'boxplot_eTIV.png')
    # plt.close()

    # Check BrainSegVolNoVent is identical to BrainSegVolNoVent.1
    if sum(freesurfer_df_banc['BrainSegVolNotVent']==freesurfer_df_banc['BrainSegVolNotVent.1'])== len(freesurfer_df_banc):
        print('BrainSegVolNotVent is identical to BrainSegVolNotVent.1')
    if sum(freesurfer_df_banc['BrainSegVolNotVent']==freesurfer_df_banc['BrainSegVolNotVent.2'])== len(freesurfer_df_banc):
        print('BrainSegVolNotVent is identical to BrainSegVolNotVent.2')

def dict_rename_columns(freesurfer_banc_columns):
    # Create a dictionary matching the columns from the ukbiobank to the BANC
    # dataset and rename ukbiobank columns.
    lh_thickness = [
                       # left hemisphere
                      'lh_thk_bankssts', 'lh_thk_caudalanteriorcingulate',
                      'lh_thk_caudalmiddlefrontal', 'lh_thk_cuneus',
                      'lh_thk_entorhinal', 'lh_thk_fusiform',
                      'lh_thk_inferiorparietal', 'lh_thk_inferiortemporal',
                      'lh_thk_isthmus', 'lh_thk_lateraloccipital',
                      'lh_thk_lateralorbitofrontal', 'lh_thk_lingual',
                      'lh_thk_medialorbitofrontal',
                      'lh_thk_middletemporal', 'lh_thk_parahippocampal',
                      'lh_thk_paracentral', 'lh_thk_parsopercularis',
                      'lh_thk_parsorbitalis', 'lh_thk_parstriangularis',
                      'lh_thk_pericalcarine', 'lh_thk_postcentral',
                      'lh_thk_posteriorcingulate', 'lh_thk_precentral',
                      'lh_thk_precuneus', 'lh_thk_rostralanteriorcingulate',
                      'lh_thk_rostralmiddlefrontal', 'lh_thk_superiorfrontal',
                      'lh_thk_superiorparietal', 'lh_thk_superiortemporal',
                      'lh_thk_supramarginal', 'lh_thk_frontalpole',
                      'lh_thk_temporalpole', 'lh_thk_transversetemporal',
                      'lh_thk_insula'
                   ]

    rh_thickness = [
                       # right hemisphere
                      'rh_thk_bankssts', 'rh_thk_caudalanteriorcingulate',
                      'rh_thk_caudalmiddlefrontal', 'rh_thk_cuneus',
                      'rh_thk_entorhinal', 'rh_thk_fusiform',
                      'rh_thk_inferiorparietal', 'rh_thk_inferiortemporal',
                      'rh_thk_isthmus', 'rh_thk_lateraloccipital',
                      'rh_thk_lateralorbitofrontal', 'rh_thk_lingual',
                      'rh_thk_medialorbitofrontal', 'rh_thk_middletemporal',
                      'rh_thk_parahippocampal', 'rh_thk_paracentral',
                      'rh_thk_parsopercularis', 'rh_thk_parsorbitalis',
                      'rh_thk_parstriangularis', 'rh_thk_pericalcarine',
                      'rh_thk_postcentral', 'rh_thk_posteriorcingulate',
                      'rh_thk_precentral', 'rh_thk_precuneus',
                      'rh_thk_rostralanteriorcingulate',
                      'rh_thk_rostralmiddlefrontal', 'rh_thk_superiorfrontal',
                      'rh_thk_superiorparietal', 'rh_thk_superiortemporal',
                      'rh_thk_supramarginal', 'rh_thk_frontalpole',
                      'rh_thk_temporalpole', 'rh_thk_transversetemporal',
                      'rh_thk_insula'
                   ]

    biobank_columns =  lh_thickness + rh_thickness +  [
                      # MaAdditional fM`eatures
                      'Left.Cerebellum.White.Matter', 'Left.Cerebellum.Cortex',
                      'Left.Thalamus.Proper', 'Left.Caudate', 'Left.Putamen',
                      'Left.Pallidum', 'X3rd.Ventricle', 'X4th.Ventricle',
                      'Brain.Stem', 'Left.Hippocampus', 'Left.Amygdala', 'CSF',
                      'Left.Accumbens.area', 'Left.VentralDC', 'Left.vessel',
                      'Right.Cerebellum.White.Matter',
                      'Right.Cerebellum.Cortex', 'Right.Thalamus.Proper',
                      'Right.Caudate', 'Right.Putamen', 'Right.Pallidum',
                      'Right.Hippocampus', 'Right.Amygdala', 'Right.Accumbens.area',
                      'Right.VentralDC', 'Right.vessel',
                      # the same
                      'CC_Posterior',
                      'CC_Mid.Posterior', 'CC_Central', 'CC_Mid_Anterior',
                      'CC_Anterior',
                      'lhCortexVol', 'rhCortexVol',
                      'CortexVol',
                      # Missing in the bionk we have
                      'lhCorticalWhiteMatterVol', 'rhCorticalWhiteMatterVol',
                      'CorticalWhiteMatterVol',
                      # 'lhCerebralWhiMateMatterVol',
                      # 'rhCerebralWhiteMatterVol', 'CerebralWhiteMatterVol',
                      'SubCortGrayVol', 'TotalGrayVol', 'SupraTentorialVol',
                      'SupraTentorialVolMaNotVent', 'SupraTentorialVolNotVentVox',
                      'MaskVol', 'BrainSegVol.to.eTIV', 'MaskVol.to.eTIV',
                      'EstimatedTotalIntraMaCranialVol'
                     ]
    # Check if you have all the features form the banc dataset
    assert(len(biobank_columns) == len(freesurfer_banc_columns))

    renameCols = dict(zip(biobank_columns, freesurfer_banc_columns))

    return lh_thickness, rh_thickness, renameCols

# Load both datasets
debug = False
resamplefactor = 1
save_path = os.path.join('/code/BayOptPy', 'freesurfer_preprocess')
project_ukbio_wd, project_data_ukbio, _ = get_paths(debug, 'UKBIO_freesurf')
project_banc_wd, project_data_banc, _  = get_paths(debug, 'BANC_freesurf')
_, _, freesurfer_df_banc = get_data(project_data_banc, 'BANC_freesurf', debug,
                                              project_banc_wd, resamplefactor,
                                              raw=True, analysis=None)
_, _, freesurfer_df_ukbio = get_data(project_data_ukbio, 'UKBIO_freesurf', debug,
                                               project_ukbio_wd, resamplefactor,
                                              raw=True, analysis=None)

# checM`k the columns between both datasets
# First Maprint the size of dataset
print('shape of the banc dataset; shape of the ukbio dataset')
print(freesurfer_df_banc.shape, freesurfer_df_ukbio.shape)
#-----------------------------------------------------------------------------
# BANC
#-----------------------------------------------------------------------------
# Check and visualise features that are only present in the BANC dataset
visualise_missing_features_banc(freesurfer_df_banc, save_path)

# remove the columns from the banc dataset that are not present in the BIOBANK
freesurfer_df_banc_clean =  drop_missing_features(freesurfer_df_banc)

# Save the lh_MeanThickness and rh_MeanThickness to compare it to calculated
# value afterwards
lh_MeanThickness_banc = freesurfer_df_banc['lh_MeanThickness_thickness']
rh_MeanThickness_banc = freesurfer_df_banc['rh_MeanThickness_thickness']

# Also drop the rh_MeanThickness_thickness and lh_MeanThickness_thickness, for
# now
freesurfer_df_banc_clean = freesurfer_df_banc_clean.drop(columns=['rh_MeanThickness_thickness',
                                                                 'lh_MeanThickness_thickness'])
# Save the colunns as variable
freesurfer_banc_columns = list(freesurfer_df_banc_clean)
freesurfer_ukbio_columns = list(freesurfer_df_ukbio)

#-----------------------------------------------------------------------------
# UKBIO
#-----------------------------------------------------------------------------
# Create mapping between the BANC and UKBIO to rename it
lh_thickness, rh_thickness, renameCols = dict_rename_columns(freesurfer_banc_columns)
freesurfer_df_ukbio.rename(columns=renameCols, inplace=True)
# Keep only the columns that both datasets have in common
df_ukbio = freesurfer_df_ukbio[freesurfer_banc_columns]
df_ukbio.index.name = 'id'
#-----------------------------------------------------------------------------
# UKBIO vs BANC
#-----------------------------------------------------------------------------
variables_of_interest = ['lhCerebralWhiteMatterVol', 'rhCerebralWhiteMatterVol',
                         'CerebralWhiteMatterVol']
# Generate a few plots to make sure you are comparing the same things among both
# datasets
df_ukbio['dataset'] = 'ukbio'
freesurfer_df_banc_clean['dataset'] = 'banc'
df_combined = pd.concat((df_ukbio, freesurfer_df_banc_clean))

for idx, variable in enumerate(variables_of_interest):
    fig = plt.figure()
    sns_boxplot = sns.boxplot(x='dataset', y=variable,
                          data=df_combined)
    plt.title(variable)
    fig = sns_boxplot.get_figure()
    fig.savefig(os.path.join(save_path, 'diagnostics', 'boxplot_%s.png'
                             %(variable)))
    plt.close()

# Calculate the Mean Thickness for both the left and the right hemisphere and
# plot the different between both datastes.
# Get the corresponding list of labels for the BANC dataset
banc_lh_thickness = [renameCols[x] for x in lh_thickness]
banc_rh_thickness = [renameCols[x] for x in rh_thickness]

df_combined['lh_MeanThickness_thickness'] = df_combined[banc_lh_thickness].mean(axis=1)
df_combined['rh_MeanThickness_thickness'] = df_combined[banc_rh_thickness].mean(axis=1)

# Create the missing value mean_thickness rh and lh for the UKBIO dataset
idx_lh_mean_thickness = freesurfer_df_banc.columns.get_loc('lh_MeanThickness_thickness')
idx_rh_mean_thickness = freesurfer_df_banc.columns.get_loc('rh_MeanThickness_thickness')
df_ukbio.insert(loc=idx_lh_mean_thickness, column='lh_MeanThickness_thickness',
               value=df_ukbio[banc_lh_thickness].mean(axis=1))
df_ukbio.insert(loc=idx_rh_mean_thickness, column='rh_MeanThickness_thickness',
               value=df_ukbio[banc_rh_thickness].mean(axis=1))

# Check if the calculated mean value is the same as the given mean value for the
# banc dataset
calculated_lh_MeanThickness_banc = \
    df_combined[df_combined['dataset'] == 'banc']['lh_MeanThickness_thickness']
calculated_rh_MeanThickness_banc = \
    df_combined[df_combined['dataset'] == 'banc']['rh_MeanThickness_thickness']

fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
plt.subplot(2, 1, 1)
plt.plot(range(len(lh_MeanThickness_banc)), calculated_lh_MeanThickness_banc -
         lh_MeanThickness_banc, marker='o', color='b')
plt.title('Left Hemisphere')
plt.subplot(2, 1, 2)
plt.plot(range(len(rh_MeanThickness_banc)), calculated_rh_MeanThickness_banc -
         rh_MeanThickness_banc, marker='o', color='r')
plt.title('Right Hemisphere')
fig.text(0.5, 0.01, 'Subject ID', ha='center')
fig.text(0.01, 0.5, 'calculated - true', va='center', rotation='vertical')
plt.tight_layout() # correct for overlaying layout
plt.savefig(os.path.join(save_path, 'diagnostics', 'MeanThickness.png'))
plt.close()

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
sns.boxplot(x='dataset', y='lh_MeanThickness_thickness',
                          data=df_combined, ax=ax1)
sns.boxplot(x='dataset', y='rh_MeanThickness_thickness',
                          data=df_combined, ax=ax2)
fig.savefig(os.path.join(save_path, 'diagnostics', 'boxplot_MeanThickness.png'))
plt.close()

# Dump the pre-processed biobank dataset that now overlaps with the features
# from the BANC dataset
save_path_ukbio = os.path.join(save_path, 'matched_dataset')
df_ukbio.to_csv(os.path.join(save_path_ukbio, 'aparc_aseg_UKB.csv'),
                columns=df_ukbio.columns)
print('Saved modifed Biobank dataset: %s' %save_path)

# As we deleted the orignal lh_MeanThickness recaculate those values
print('Save modifed verion of BANC freesurfer where the MeanThickness was \
      calculated')
freesurfer_df_banc_clean.insert(loc=idx_lh_mean_thickness, column='lh_MeanThickness_thickness',
               value=freesurfer_df_banc_clean[banc_lh_thickness].mean(axis=1))
freesurfer_df_banc_clean.insert(loc=idx_rh_mean_thickness, column='rh_MeanThickness_thickness',
               value=freesurfer_df_banc_clean[banc_rh_thickness].mean(axis=1))
save_path_banc = os.path.join(save_path, 'matched_dataset')
freesurfer_df_banc_clean.to_csv(os.path.join(save_path_banc, 'aparc_aseg_BANC.csv'),
                                columns=freesurfer_df_banc_clean.columns)

# Check that both save dataset have the same number of columns
assert(len(df_ukbio.columns) == len(freesurfer_df_banc_clean.columns))

#-----------------------------------------------------------------------------
# Select only summary features from  UKBIO and BANC
#-----------------------------------------------------------------------------
# Create another dataframe where you only have summary features from both
# datasets
summary_columns =  ['Left-Cerebellum-White-Matter',
                   'Left-Cerebellum-Cortex',
                   'CSF',
                   'Left-VentralDC',
                   'Right-Cerebellum-White-Matter',
                   'Right-Cerebellum-Cortex',
                   'Right-VentralDC',
                   'lhCortexVol',
                   'rhCortexVol',
                   'CortexVol',
                   'lhCerebralWhiteMatterVol',
                   'rhCerebralWhiteMatterVol',
                   'CerebralWhiteMatterVol',
                   'SubCortGrayVol',
                   'TotalGrayVol',
                   'SupraTentorialVol',
                   'SupraTentorialVolNotVent',
                   'MaskVol',
                   'BrainSegVol-to-eTIV',
                   'MaskVol-to-eTIV',
                   'EstimatedTotalIntraCranialVol',
                   'dataset']

summary_df_banc = freesurfer_df_banc_clean[summary_columns]
summary_df_banc.to_csv(os.path.join(save_path_banc,
                        'aparc_aseg_BANC_summary.csv'),
                        index_label='BANC_ID')
print('Saved summary banc dataset: %s' %save_path_banc)
pairplot = sns.pairplot(summary_df_banc)
pairplot.savefig(os.path.join(save_path_banc, 'pairplot_banc.png'), format='png')

summary_df_ukbio = df_ukbio[summary_columns]
summary_df_ukbio.to_csv(os.path.join(save_path_ukbio,
                       'aparc_aseg_UKBIO_summary.csv'),
                       index_label='id')
print('Saved summary ukbio dataset: %s' %save_path_banc)
pairplot = sns.pairplot(summary_df_ukbio)
pairplot.savefig(os.path.join(save_path_banc, 'pairplot_ukbio.png'), format='png')
print('I am done!')
# Compare the columns entry
