import joblib
import os
import shutil
import re
import pandas as pd
from multiprocessing import Process, Pool
from functools import partial
from nilearn import masking, image
import nibabel as nib
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def get_paths(debug, dataset):

    if debug and dataset == 'OASIS':
        project_wd = os.getcwd()
        project_data = os.path.join(project_wd, 'data')
        project_sink = os.path.join(project_data, 'output')
    elif debug and dataset == 'BANC':
        project_wd = os.getcwd()
        project_data = os.path.join(os.getenv('HOME'), 'NaN', 'BANC_2016')
        project_sink = os.path.join(project_data, 'output')
    elif debug and dataset == 'BOSTON':
        project_wd = os.getcwd()
        project_data = None
        project_sink = None
    elif debug and dataset == 'BANC_freesurf':
        project_wd = os.getcwd()
        project_data = os.path.join(os.getenv('HOME'), 'NaN', 'BANC_2016')
        project_sink = None
    elif debug and dataset == 'UKBIO_freesurf':
        project_wd = os.getcwd()
        project_data = os.path.join('UKBIO')
        project_sink = None
    elif not debug and dataset == 'OASIS':
        project_wd = '/code'
        project_data = os.path.join(os.sep, 'NaN', 'data')
        project_sink = os.path.join(project_data, 'output')
    elif not debug and dataset == 'BANC':
        project_wd = '/code'
        project_data = os.path.join(os.sep, 'data', 'NaN', 'BANC_2016')
        project_sink = os.path.join(project_data, 'output')
    elif not debug and dataset == 'BOSTON':
        project_wd = '/code'
        project_data = None
        project_sink = None
    elif not debug and dataset == 'BANC_freesurf':
        project_wd = '/code'
        project_data = os.path.join(os.sep, 'data', 'NaN', 'BANC_2016')
        project_sink = None
    elif not debug and dataset == 'UKBIO_freesurf':
        project_wd = '/code'
        project_data = os.path.join(os.sep, 'code', 'UKBIO')
        project_sink = None

    else:
        raise ValueError('Analysis for this dataset is not yet implemented!')

    print('Code Path: %s' %project_wd)
    print('Data Path: %s' %project_data)
    print('Data Out: %s' %project_sink )
    return project_wd, project_data, project_sink

def get_output_path(analysis, ngen, random_seed, population_size, debug,
                    mutation, crossover):
    # Check if output path exists, otherwise create it
    rnd_seed_path = get_all_random_seed_paths(analysis, ngen, population_size,
                                              debug, mutation, crossover)
    output_path = os.path.join(rnd_seed_path, 'random_seed_%03d' %random_seed)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    return output_path

def get_all_random_seed_paths(analysis, ngen, population_size, debug, mutation,
                             crossover):
    # As they should have been created by the get_output_path, do not create
    # path but just find its location
    if analysis == 'vanilla' or analysis == 'feat_selec' or \
        analysis == 'feat_combi' or analysis == 'vanilla_combi' or \
        analysis == 'random_seed' or analysis == 'ukbio':
        if debug:
            output_path = os.path.join('BayOptPy', 'tpot', analysis,
                                       '%03d_generations' %ngen)
        else:
            output_path = os.path.join(os.sep, 'code', 'BayOptPy', 'tpot', analysis,
                                       '%03d_generations' %ngen)
    elif analysis == 'population':
        if debug:
            output_path = os.path.join('BayOptPy', 'tpot', analysis,
                                       '%05d_population_size' %population_size,
                                       '%03d_generations' %ngen)
        else:
            output_path = os.path.join(os.sep, 'code', 'BayOptPy', 'tpot', analysis,
                                       '%05d_population_size' %population_size,
                                       '%03d_generations' %ngen)
    elif analysis == 'mutation':
        if debug:
            output_path = os.path.join('BayOptPy', 'tpot', analysis,
                                       '%03d_generations' %ngen,
                                       '%.01f_mut_%.01f_cross' %(mutation, crossover))
        else:
            output_path = os.path.join(os.sep, 'code', 'BayOptPy', 'tpot', analysis,
                                       '%03d_generations' %ngen,
                                       '%.01f_mut_%.01f_cross' %(mutation, crossover))

    else:
        raise IOError('Analysis path not defined. Passed analysis was %s'
                      %analysis)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    return output_path

def get_best_pipeline_paths(analysis, ngen, random_seed, population_size, debug,
                           mutation, crossover):
    # check if folder exists and in case yes, remove it as new runs will save
    # new files without overwritting
    output_path = get_output_path(analysis, ngen, random_seed, population_size,
                                  debug, mutation, crossover)
    checkpoint_path = os.path.join(output_path, 'checkpoint_folder')

    # Delete folder if it already exists and create a new one
    if os.path.exists(checkpoint_path):
        shutil.rmtree(checkpoint_path)
        print('Deleted pre-exiting checkpoint folder')

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
        print('Creating checkpoint folder')

    return checkpoint_path

def drop_missing_features(dataframe):
    '''
    This function takes a dataframe and removes the already defined missing
    columns from the dataframe.
    '''
    missing_features = ['BrainSegVolNotVent',
                        'eTIV',
                        'BrainSegVolNotVent.1', 'eTIV.1']
    cleaned_df = dataframe.drop(missing_features, axis=1)
    return cleaned_df

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

    # do some sanity checks
    # Check if you have the same number of selectedsubjectsid as the demographic information
    assert(len(selectedSubId) == len(demographics))

    return demographics, selectedSubId


def _multiprocessing_resample(img, target_affine):
    resampled_img = image.resample_img(img, target_affine=target_affine,
                                       interpolation='nearest')
    return resampled_img


def _load_nibabel(filePath):
    img = nib.load(os.path.join(filePath))
    return img

def get_config_dictionary():
    # Define the same default pipeline as TPOT light but without the preprocessing operators
    regressor_config_dic = {

        'sklearn.linear_model.ElasticNetCV': {
            'l1_ratio': np.arange(0.0, 1.01, 0.05),
            'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        },

        'sklearn.tree.DecisionTreeRegressor': {
            'max_depth': range(1, 11),
            'min_samples_split': range(2, 21),
            'min_samples_leaf': range(1, 21)
        },

        'sklearn.neighbors.KNeighborsRegressor': {
            'n_neighbors': range(1, 101),
            'weights': ["uniform", "distance"],
            'p': [1, 2]
        },

        'sklearn.linear_model.LassoLarsCV': {
            'normalize': [True, False]
        },

        'sklearn.svm.LinearSVR': {
            'loss': ["epsilon_insensitive", "squared_epsilon_insensitive"],
            'dual': [True, False],
            'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
            'epsilon': [1e-4, 1e-3, 1e-2, 1e-1, 1.]
        },

        'sklearn.linear_model.RidgeCV': {
        },

        # Selectors
        'sklearn.feature_selection.SelectFwe': {
            'alpha': np.arange(0, 0.05, 0.001),
            'score_func': {
                'sklearn.feature_selection.f_regression': None
            }
        },

        'sklearn.feature_selection.SelectPercentile': {
            'percentile': range(1, 100),
            'score_func': {
                'sklearn.feature_selection.f_regression': None
            }
        },

        'sklearn.feature_selection.VarianceThreshold': {
            'threshold': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
        }

    }
    return regressor_config_dic

def get_mean_age(df):
    mean_age = df['Age'].mean()
    std_age = df['Age'].std()
    print('Mean Age %.2f +- %.2f' %(mean_age, std_age))

def get_data(project_data, dataset, debug, project_wd, resamplefactor):
    ''' Load the csv files and return
    :param project_data:
    :param dataset:
    :param debug:
    :param project_wd:
    :param resamplefactor:
    :return: demographics:
    :return: demographics:
    :return: dataframe.values: Just the numeric values of the dataframe
    '''
    print('Loading Brain image data')
    if dataset == 'OASIS':
        # remove the file end and get list of all used subjects
        fileList = os.listdir(project_data)
        rawsubjectsId = [re.sub(r'^smwc1(.*?)\_mpr-1_anon.nii$', '\\1', file) for file in fileList if file.endswith('.nii')]
        # TODO: Change this. For testing purpose select just the first 5 subjects
        #rawsubjectsId = rawsubjectsId[:25]

        # Load the demographics for each subject
        demographics, selectedSubId = get_data_covariates(project_data, rawsubjectsId, dataset)
        # print subjects mean age
        get_mean_age(demographics)

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
        # print subjects mean age
        get_mean_age(demographics)
        # Get the file path of the selected subjects
        subjectsFile = [os.path.join(project_data_path, file) for file in fileList if file[5:12] in selectedSubId]

        # Load image proxies
        with Pool() as p:
            imgs = list(tqdm(p.imap(_load_nibabel, subjectsFile), total=len(selectedSubId)))

    elif dataset == 'BANC_freesurf':
        freesurf_df = pd.read_csv(os.path.join(project_wd, 'BayOptPy',
                                               'freesurfer_preprocess',
                                               'original_dataset',
                                               'BANC',
                                               'aparc_aseg_stats_BANC.csv'), delimiter=',', index_col=0)
        rawsubjectsId = freesurf_df.index

        # Load the demographics for each subject
        demographics, selectedSubId = get_data_covariates(project_data, rawsubjectsId, 'BANC')
        # return numpy array of the dataframe
        return demographics, None, freesurf_df

    elif dataset == 'UKBIO_freesurf':
        freesurf_df = pd.read_csv(os.path.join(project_wd, 'BayOptPy',
                                               'freesurfer_preprocess',
                                               'original_dataset',
                                               'UKBIO', 'UKB_10k_FS_4844.csv'), delimiter=',')
        return None, None, freesurf_df

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

def get_mae_for_all_generations(dataset, random_seed, generations, config_dict,
                               tpot_path):
    '''
    Get the MAE values for both the training and test dataset
    :return:
    '''
    # Load the scores for the best models
    saved_path = os.path.join(tpot_path, 'random_seed_%03d' %random_seed,
                                   'tpot_%s_%s_%03dgen_pipelines.dump'
                                    %(dataset, config_dict, generations))
    # Note that if a value is not present for a generation, that means that the
    # score did not change from the previous generation
    # sort the array in ascending order
    logbook = joblib.load(saved_path)
    gen = list(logbook['log'].keys())

    print('There are %d optminal pipelines' %len(gen))
    print('These are the best pipelines')
    for generation in gen:
        print(logbook['log'][generation]['pipeline_name'])

    # Iterate over the the list of saved MAEs and repeat the values where one
    # generation is missed
    all_mae_test = []
    all_mae_train = []
    pipeline_complexity = []
    curr_gen_idx = 0
    # all generations
    for generation in range(generations):
        if generation == gen[curr_gen_idx]:
            all_mae_test.append(abs(logbook['log'][gen[curr_gen_idx]]['pipeline_test_mae']))
            all_mae_train.append(abs(logbook['log'][gen[curr_gen_idx]]['pipeline_score']))
            pipeline_complexity.append(len(logbook['log'][gen[curr_gen_idx]]['pipeline_sklearn_obj'].named_steps.keys()))
            if len(gen) > 1 and (len(gen) > curr_gen_idx + 1):
                curr_gen_idx += 1
        else:
            # repeat the same last value
            all_mae_test.append(all_mae_test[-1])
            all_mae_train.append(all_mae_train[-1])
            pipeline_complexity.append(pipeline_complexity[-1])

    # transform the pipeline_complexity into a numpy array, in order to perform
    # fancy indexing
    pipeline_complexity = np.array(pipeline_complexity)
    return all_mae_test, all_mae_train, pipeline_complexity

def set_publication_style():
    # Se font size to paper size
    plt.style.use(['seaborn-white', 'seaborn-talk'])
    matplotlib.rc("font", family="Times New Roman")
    # Remove the spines
    sns.set_style('white', {"axes.spines.top": False,
                            "axes.spines.right": False,
                            "axes.labelsize": 'large'})

def create_age_histogram(training_age, test_age, dataset):
    '''
    Get an age array and plot and save the age histogram for the analysed sample
    '''
    # Define plot styple
    set_publication_style()
    if dataset == 'BANC':
        path_to_save = '/code/BayOptPy/tpot/age_histogram_BANC.png'
        # Define plot range
        # 17 and 89 are the min and max age on the dataset respectively
        min_age = 17
        max_age = 89
        plt.hist(training_age, bins=65, range=(min_age,max_age), label='training')
        plt.hist(test_age, bins=65, range=(min_age,max_age), label='test')
    elif dataset == 'UKBIO':
        path_to_save = '/code/UKBIO/age_histogram_UKBIO.png'
        min_age = training_age.min()
        max_age = training_age.max()
        plt.hist(training_age, bins=int(max_age-min_age),
                 range=(min_age,max_age), color='#82E0AA')
    plt.xlabel('Age')
    plt.ylabel('# of Subjects')
    plt.legend()
    plt.savefig(path_to_save)
    plt.close()
