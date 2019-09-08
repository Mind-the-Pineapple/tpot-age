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
    random_seeds = np.arange(10, 110+10, 10)

    def tpot_model_analysis(random_seed, save_path):
        save_path = os.path.join(save_path, 'random_seed_%03d' %random_seed)
        print('Random seed: %03d' %random_seed)
        # Load the clean data for both the UKBIO and the BANC analysis
        # This version of the UKBIOBANK dataset contains the same columns as the BANC
        # dataset

        # Load the saved trained model
        tpot = joblib.load(os.path.join(save_path, 'tpot_%s_%s_%03dgen.dump'
                                        %(args.dataset, args.analysis, args.generations)))
        exported_pipeline = tpot['fitted_pipeline']

        # Load the saved validation dataset
        project_ukbio_wd, project_data_ukbio, _ = get_paths(args.debug, args.dataset)
        with open(os.path.join(save_path, 'splitted_dataset_%s.pickle' %args.dataset), 'rb') as handle:
            splitted_dataset = pickle.load(handle)

        # Print some results
        print('Print MAE - test')
        y_predicted_test = exported_pipeline.predict(splitted_dataset['Xtest_scaled'])
        mae_test = mean_absolute_error(splitted_dataset['Ytest'], y_predicted_test)
        print(mae_test)
        print('Print MAE - training')
        y_predicted_train = exported_pipeline.predict(splitted_dataset['Xtrain_scaled'])
        mae_train = mean_absolute_error(splitted_dataset['Ytrain'], y_predicted_train)
        print(mae_train)
        print('Print MAE - validation')
        y_predicted_validation = exported_pipeline.predict(splitted_dataset['Xvalidate_scaled'])
        mae_validation = mean_absolute_error(splitted_dataset['Yvalidate'], y_predicted_validation)
        print(mae_validation)

        # plot predicted vs true for the test
        output_path_test = os.path.join(save_path, 'test_predicted_true_age.eps')
        plot_predicted_vs_true(splitted_dataset['Ytest'], y_predicted_test,
                                   output_path_test, 'Age')

        # plot predicted vs true for the validation
        output_path_val = os.path.join(save_path, 'validation_predicted_true_age.eps')
        plot_predicted_vs_true(splitted_dataset['Yvalidate'],
                                   y_predicted_validation, output_path_val, 'Age')

        # Do some statistics. Calculate R2 and the Spearman
        from scipy.stats import spearmanr, pearsonr
        from sklearn.metrics import r2_score

        rho_val, rho_p_value_val = spearmanr(splitted_dataset['Yvalidate'],
                                     y_predicted_validation)
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

        print('Statistics for the full validation dataset')
        print('shape of the dataset: %s' %(splitted_dataset['Yvalidate'].shape,))
        print('Rho and p-value: %.4f %.4f' %(rho_val, rho_p_value_val))

        r_score_val = r2_score(splitted_dataset['Yvalidate'], y_predicted_validation)
        print('R2 is: %.4f' %r_score_val)

        r_val, r_p_value_val = pearsonr(splitted_dataset['Yvalidate'], y_predicted_validation)
        print('R is: %.4f' %r_val)

        #-----------------------------------------------------------------------------
        # Use just part of the validation dataset
        from sklearn.model_selection import train_test_split

        X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(
                         splitted_dataset['Xvalidate_scaled'],
                         splitted_dataset['Yvalidate'],
                         test_size=.4, random_state=42)

        predicted_subset_val = exported_pipeline.predict(X_train_val)
        rho_subset_val, rho_p_value_subset_val = spearmanr(y_train_val,
                                                          predicted_subset_val)

        r_subset_val, r_p_value_subset_val = pearsonr(y_train_val,
                                                      predicted_subset_val)
        print('')
        print('Statistics on part of the dataset')
        print('shape of the dataset: %s' %(X_train_val.shape,))
        print('Rho and p-value: %.4f %.4f' %(rho_subset_val, rho_p_value_subset_val))
        print('R is: %.4f' %r_subset_val)

        # plot predicted vs true for the test
        output_path_test = os.path.join(save_path,
                                        'subset_val_predicted_true_age.eps')
        plot_predicted_vs_true(y_train_val, predicted_subset_val,
                                   output_path_test, 'Age')
        #-----------------------------------------------------------------------------
        # Save count of the number of models
            # Define the list of possible models
        algorithms_list = ['GaussianProcessRegressor', 'RVR',
                           'LinearSVR',
                            'RandomForestRegressor',
                            'KNeighborsRegressor',
                            'LinearRegression',
                            'Ridge','ElasticNetCV',
                            'ExtraTreesRegressor',
                            'LassoLarsCV',
                            'DecisionTreeRegressor']

        algorithms_count = dict.fromkeys(algorithms_list, 0)
        # Trasform the pipeline into a string and search for patterns
        values = str(exported_pipeline.named_steps.values())
        for algorithm in algorithms_list:
            algorithms_count[algorithm] += values.count(algorithm + '(')
        algorithms_count['random_seed'] = random_seed
        print('-------------------------------------------------------------------------------')
        return mae_test, mae_validation, r_val, r_test, algorithms_count


    if args.analysis == 'mutation':
        save_path = '/code/BayOptPy/tpot_regression/Output/%s/age/%03d_generations/%s_mut_%s_cross'  \
                %(args.analysis, args.generations, args.mutation_rate,
                  args.crossover_rate)
    else:
        save_path = '/code/BayOptPy/tpot_regression/Output/%s/age/%03d_generations/' %(args.analysis, args.generations)


    mae_test_all = []
    mae_validation_all = []
    r_val_all = []
    r_test_all = []
    algorithms_count_all = []
    for random_seed in random_seeds:
        mae_test, mae_validation, r_val, r_test, algorithms_count = tpot_model_analysis(random_seed, save_path)
        mae_test_all.append(mae_test)
        mae_validation_all.append(mae_validation)
        r_val_all.append(r_val)
        r_test_all.append(r_test)
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

    # Plot MAE across all random seeds
    plt.figure()
    ind = np.arange(1)
    plt.bar(ind, np.mean(mae_test_all), yerr=[np.std(mae_test_all)])
    plt.xticks(ind, (args.analysis))
    plt.savefig(os.path.join(save_path, 'MAE_%s_bootsraped.eps' %args.analysis))

    print('Mean and std for test data')
    print(np.mean(mae_test_all), np.std(mae_test_all))
    print('Mean and std for validation data')
    print(np.mean(mae_validation_all), np.std(mae_validation_all))
    print('Mean and std pearson corr test data')
    print(np.mean(r_test_all), np.std(r_test_all))
    print('Mean and std pearson corr validation data')
    print(np.mean(r_val_all), np.std(r_val_all))

    results = {'mae_test': mae_test_all,
               'mae_validation': mae_validation_all,
               'r_val': r_val_all,
               'r_test': r_test_all}
    with open(os.path.join(save_path, 'tpot_all_seeds.pckl'), 'wb') as handle:
        pickle.dump(results, handle)




