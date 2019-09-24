import argparse
import os
import pickle

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import ElasticNetCV
from tpot.builtins import StackingEstimator

from BayOptPy.helperfunctions import (get_paths, get_data,
                                      drop_missing_features,
                                      set_publication_style,
                                      plot_predicted_vs_true)

"""
BANC + TPOT dataset
This script tests the best model recommened by TPOT for 100 generations for
multiple random seeds
"""

parser = argparse.ArgumentParser()
set_publication_style()
parser.add_argument('-debug',
                    dest='debug',
                    action='store_true',
                    help='Run debug with Pycharm'
                    )
parser.add_argument('-model',
                    dest='model',
                    help='Define if a classification or regression problem',
                    choices=['regression', 'classification', 'classification2']
                    )
parser.add_argument('-dataset',
                    dest='dataset',
                    help='Specify which dataset to use',
                    choices=['OASIS',            # Images from OASIS
                             'BANC',             # Images from BANC
                             'BANC_freesurf',    # Freesurfer info from BANC
                             'freesurf_combined', # Use Freesurfer from BANC and
                                                 # UKBIO
                             'UKBIO_freesurf']
                    )
parser.add_argument('-generations',
                    dest='generations',
                    help='Specify number of generations to use',
                    type=int,
                    required=True
                    )
parser.add_argument('-resamplefactor',
                    dest='resamplefactor',
                    help='Specify resampling rate for the image affine',
                    type=int,
                    default=1 # no resampling is performed
                    )
parser.add_argument('-analysis',
                    dest='analysis',
                    help='Specify which type of analysis to use',
                    choices=['vanilla', 'population', 'feat_selec',
                             'feat_combi', 'vanilla_combi', 'mutation',
                             'random_seed', 'ukbio', 'summary_data'],
                    required=True
                    )
parser.add_argument('-mutation_rate',
                   dest='mutation_rate',
                   help='Must be on the range [0, 1.0]',
                   type=float,
                   default=.9
                   )
parser.add_argument('-crossover_rate',
                    dest='crossover_rate',
                    help='Cross over of the genetic algorithm. Must be on \
                    the range [0, 1.0]',
                    type=float,
                    default=.1
                   )
args = parser.parse_args()

if __name__ == '__main__':
    # General Settings
    #-------------------------------------------------------------------------------
    random_seeds = np.arange(10, 110, 10)

    def tpot_model_analysis(random_seed, save_path):
        save_path = os.path.join(save_path, 'random_seed_%03d' %random_seed)
        print('Random seed: %03d' %random_seed)
        # Load the clean data for both the UKBIO and the BANC analysis
        # This version of the UKBIOBANK dataset contains the same columns as the BANC
        # dataset

        # Load the saved trained model
        tpot = joblib.load(os.path.join(save_path, 'tpot_%s_vanilla_combi_%03dgen.dump'
                                        %(args.dataset, args.generations)))
        exported_pipeline = tpot['fitted_pipeline']

        # Load the saved test dataset
        project_ukbio_wd, project_data_ukbio, _ = get_paths(args.debug, args.dataset)
        with open(os.path.join(save_path, 'splitted_dataset_%s.pickle' %args.dataset), 'rb') as handle:
            splitted_dataset = pickle.load(handle)

        # Print some results
        print('Print MAE - test')
        y_predicted_test = exported_pipeline.predict(splitted_dataset['Xtest_scaled'])
        mae_test = mean_absolute_error(splitted_dataset['Ytest'], y_predicted_test)
        print(mae_test)
        print('Print MAE - training')
        y_predicted_train = exported_pipeline.predict(splitted_dataset['Xtrain_scaled'])
        mae_train = mean_absolute_error(splitted_dataset['Ytrain'], y_predicted_train)
        print(mae_train)

        # plot predicted vs true for the test
        output_path_test = os.path.join(save_path, 'test_predicted_true_age.eps')
        plot_predicted_vs_true(splitted_dataset['Ytest'], y_predicted_test,
                                   output_path_test, 'Age')


        # Do some statistics. Calculate R2 and the Spearman
        from scipy.stats import spearmanr, pearsonr
        from sklearn.metrics import r2_score

        rho_test, rho_p_value_test = spearmanr(splitted_dataset['Ytest'],
                                     y_predicted_test)
        print('Statistics for the test dataset')
        print('shape of the dataset: %s' %(splitted_dataset['Ytest'].shape,))
        print('Rho and p-value: %.4f %.4f' %(rho_test, rho_p_value_test))

        r_score_test = r2_score(splitted_dataset['Ytest'], y_predicted_test)
        print('R2 is: %.4f' %r_score_test)

        r_test, r_p_value_test = pearsonr(splitted_dataset['Ytest'],
                                              y_predicted_test)
        print('R is: %.4f' %r_test)
        return mae_test, r_test


    if args.analysis == 'mutation':
        save_path =
        '/code/BayOptPy/tpot_regression/Output/%s/age/%03d_generations/%s_mut_%s_cross/simple_ttest'  \
                %(args.analysis, args.generations, args.mutation_rate,
                  args.crossover_rate)
    else:
        save_path =
        '/code/BayOptPy/tpot_regression/Output/%s/age/%03d_generations/simple_ttest/' %(args.analysis, args.generations)

    # check if save path exists otherwise create it
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    mae_test_all = []
    r_test_all = []
    for random_seed in random_seeds:
        mae_test, r_test = tpot_model_analysis(random_seed, save_path)
        mae_test_all.append(mae_test)
        r_test_all.append(r_test)

    print('Mean and std for test data')
    print(np.mean(mae_test_all), np.std(mae_test_all))
    print('Mean and std pearson corr test data')
    print(np.mean(r_test_all), np.std(r_test_all))

    results = {'mae_test': mae_test_all,
               'r_test': r_test_all}
    with open(os.path.join(save_path, 'tpot_all_seeds.pckl'), 'wb') as handle:
        pickle.dump(results, handle)



