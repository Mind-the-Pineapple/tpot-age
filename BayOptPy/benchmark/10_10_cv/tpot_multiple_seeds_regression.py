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
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import make_pipeline, make_union
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import ElasticNetCV
from tpot.builtins import StackingEstimator
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score

from BayOptPy.helperfunctions import (get_paths, get_data,
                                      drop_missing_features,
                                      set_publication_style,
                                      plot_predicted_vs_true,
                                      ttest_ind_corrected)

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
parser.add_argument('-config_dic',
                    dest='config_dic',
                    help='Specify which type of config_dic to use',
                    choices=['vanilla', 'feat_selec',
                             'feat_combi', 'vanilla_combi'],
                    required=True
                    )
parser.add_argument('-analysis',
                    dest='analysis',
                    help='Specify which type of analysis to use',
                    choices=['vanilla', 'population', 'feat_selec',
                             'feat_combi', 'vanilla_combi', 'mutation',
                             'random_seed', 'ukbio', 'summary_data',
                             'uniform_dist'],
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
    # Analysed random seeds
    min_repetition = 10
    max_repetition = 210
    step_repetition = 10
    random_seeds = np.arange(min_repetition, max_repetition+step_repetition,
                             step_repetition)

    def tpot_model_analysis(random_seed, save_path, n_folds, algorithms_list):
        save_path = os.path.join(save_path, 'random_seed_%03d' %random_seed)
        print('Random seed: %03d' %random_seed)
        # Load the clean data for both the UKBIO and the BANC analysis
        # This version of the UKBIOBANK dataset contains the same columns as the BANC
        # dataset

        # Load the saved trained model
        tpot = joblib.load(os.path.join(save_path, 'tpot_%s_%s_%03dgen.dump'
                                        %(args.dataset, args.config_dic, args.generations)))
        exported_pipeline = tpot['fitted_pipeline']

        # Load the saved validatVion dataset
        project_ukbio_wd, project_data_ukbio, _ = get_paths(args.debug, args.dataset)
        with open(os.path.join(save_path, 'splitted_dataset_%s.pickle' %args.dataset), 'rb') as handle:
            splitted_dataset = pickle.load(handle)

        # Set target and features
        x = splitted_dataset['Xtest_scaled']
        y = splitted_dataset['Ytest']

        # # Test on the entire dataset and see if the value is similar to K-Fold
        # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.4,
        #                                                    random_state=random_seed)
        # new_new_model = exported_pipeline.fit(x_train, y_train)
        # new_y_predicted = new_new_model.predict(x_test)
        # new_acc = mean_absolute_error(y_test, new_y_predicted)
        # print('MAE on entire dataset')
        # print(new_acc)

        # Perform KFold analysis
        kf = KFold(n_splits=n_folds, random_state=random_seed)
        mae_cv = np.zeros((n_folds, 1))
        pearsons_corr = np.zeros((n_folds, 1))
        pearsons_pval = np.zeros((n_folds, 1))

        for i_fold, (train_idx, test_idx) in enumerate(kf.split(x, y)):
            x_train, x_test = x[train_idx, :], x[test_idx, :]
            y_train, y_test = y[train_idx], y[test_idx]

            print('CV iteration: %d' %(i_fold + 1))
            print('Shape of the trainig and test dataset')
            print(y_train.shape, y_test.shape)

            # train the model
            new_model = exported_pipeline.fit(x_train, y_train)

            # test the model
            y_predicted = new_model.predict(x_test)
            mae_kfold = mean_absolute_error(y_test, y_predicted)
            mae_cv[i_fold, :] = mae_kfold
            # now look at the pearson's correlation
            r_test, r_p_value_test = pearsonr(y_test, y_predicted)
            pearsons_corr[i_fold, :] = r_test
            pearsons_pval[i_fold, :] = r_p_value_test

        print('CV results')
        print('MAE: Mean(SD) = %.3f(%.3f)' % (mae_cv.mean(), mae_cv.std()))
        print('Pearson\'s Correlation: Mean(SD) = %.3f(%.3f)' % (r_test.mean(),
                                                                 r_test.std()))

        # plot predicted vs true for the test (Entire sample)
        print('Plotting Predicted Vs True Age for all the sample')
        output_path_test = os.path.join(save_path,
                                        'test_predicted_true_age_rnd_seed%d.eps'
                                       %random_seed)
        y_predicted_test = exported_pipeline.predict(splitted_dataset['Xtest_scaled'])
        plot_predicted_vs_true(splitted_dataset['Ytest'], y_predicted_test,
                                   output_path_test, 'Age')

        #-----------------------------------------------------------------------------
        # Save count of the number of models

        algorithms_count = dict.fromkeys(algorithms_list, 0)
        # Trasform the pipeline into a string and search for patterns
        values = str(exported_pipeline.named_steps.values())
        for algorithm in algorithms_list:
            algorithms_count[algorithm] += values.count(algorithm + '(')
        algorithms_count['random_seed'] = random_seed
        print('-------------------------------------------------------------------------------')
        return mae_cv, pearsons_corr, algorithms_count


    if args.analysis == 'mutation':
        save_path = '/code/BayOptPy/tpot_regression/Output/%s/age/%03d_generations/%s_mut_%s_cross'  \
                %(args.analysis, args.generations, args.mutation_rate,
                  args.crossover_rate)
    else:
        save_path = '/code/BayOptPy/tpot_regression/Output/%s/age/%03d_generations/' %(args.analysis, args.generations)

    # Define the list of possible models
    algorithms_list = ['GaussianProcessRegressor',
                       'RVR',
                       'LinearSVR',
                       'RandomForestRegressor',
                       'KNeighborsRegressor',
                       'LinearRegression',
                       'Ridge','ElasticNetCV',
                       'ExtraTreesRegressor',
                       'LassoLarsCV',
                       'DecisionTreeRegressor']
    n_folds = 10
    mae_test_all = np.zeros((len(random_seeds), n_folds))
    r_test_all = np.zeros((len(random_seeds), n_folds))
    algorithms_count_all = []
    for seed_idx, random_seed in enumerate(random_seeds):
        mae_test, r_test, algorithms_count = tpot_model_analysis(random_seed,
                                                                 save_path,
                                                                 n_folds,
                                                                 algorithms_list)
        mae_test_all[seed_idx, :] = mae_test.T
        r_test_all[seed_idx, :] = r_test.T
        algorithms_count_all.append(algorithms_count)

    # Transfrom algorithm counts into a dataframe and plot heatmap
    df = pd.DataFrame(algorithms_count_all)
    df = df.set_index('random_seed')
    df = df.transpose()
    plt.figure(figsize=(28,8))
    sns.heatmap(df, cmap='YlGnBu')
    plt.xlabel('Random Seeds')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'algorithms_count.eps'))
    plt.close()

    # Print the sum information
    print('Total count for each model:')
    print(df.sum(axis=1))

    # Plot MAE across all random seeds
    plt.figure()
    ind = np.arange(1)
    plt.bar(ind, np.mean(np.mean(mae_test_all, axis=0)),
                 yerr=[np.std(np.std(mae_test_all, axis=0))])
    plt.xticks(ind, (args.analysis))
    plt.savefig(os.path.join(save_path, 'MAE_%s_bootsraped.eps' %args.analysis))

    print('Mean and std for test data')
    print(np.mean(mae_test_all, axis=0), np.std(mae_test_all, axis=0))
    print('Mean and std pearson corr test data')
    print(np.mean(r_test_all, axis=0), np.std(r_test_all, axis=0))

    results = {'mae_test': mae_test_all,
               'r_test': r_test_all}
    with open(os.path.join(save_path, 'tpot_all_seeds.pckl'), 'wb') as handle:
        pickle.dump(results, handle)




