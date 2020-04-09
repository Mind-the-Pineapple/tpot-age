import os
import argparse
import joblib
import pickle
import numpy as np
import time
from pathlib import Path

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
from skrvm import RVR
from scipy.stats import spearmanr, pearsonr

from BayOptPy.helperfunctions import (get_paths, plot_predicted_vs_true,
                                      set_publication_style)


parser = argparse.ArgumentParser()
parser.add_argument('-analysis',
                    dest='analysis',
                    help='Specify which type of analysis to use',
                    choices=['vanilla_combi',
                             'uniform_dist'],
                    required=True
                    )
args = parser.parse_args()

def rvr_analysis(random_seed, save_path, n_folds, analysis):
    save_path = save_path / ('random_seed_%03d' %random_seed)
    print('Random seed: %03d' %random_seed)
    # Load the saved validation dataset
    project_ukbio_wd, project_data_ukbio, _ = get_paths(debug, dataset)
    with open(save_path / ('splitted_dataset_%s.pickle' %dataset), 'rb') as handle:
            splitted_dataset = pickle.load(handle)

    kf = KFold(n_splits=n_folds, random_state=random_seed)
    mae_cv = np.zeros((n_folds, 1))
    pearsons_corr = np.zeros((n_folds, 1))
    pearsons_pval = np.zeros((n_folds, 1))

    # Set target and features
    x = splitted_dataset['Xtest_scaled']
    y = splitted_dataset['Ytest']

    t_time_train = []
    t_time_test = []

    for i_fold, (train_idx, test_idx) in enumerate(kf.split(x, y)):
        x_train, x_test = x[train_idx, :], x[test_idx, :]
        y_train, y_test = y[train_idx], y[test_idx]

        print('CV iteration: %d' %(i_fold + 1))
        print('Shape of the trainig and test dataset')
        print(y_train.shape, y_test.shape)

        # train the model
        model = RVR(kernel='linear')
        cv_time_train = time.process_time()
        model.fit(x_train, y_train)
        elapsed_time = time.process_time() - cv_time_train
        print('CV - Elapased time in seconds to train:')
        t_time_train.append(elapsed_time)
        print('%.03f' %elapsed_time)

        # test the model
        cv_time_test = time.process_time()
        y_predicted = model.predict(x_test)
        elapsed_time = time.process_time() - cv_time_test
        t_time_test.append(elapsed_time)
        print('CV - Elapased time in seconds to test:')
        print('%.03f' %elapsed_time)

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
    print('Mean CV time: %.3f s ' %np.mean(t_time_train))
    print('SD CV time: %.3f s' %np.std(t_time_train))
    print('Mean CV time: %.3f s ' %np.mean(t_time_test))
    print('SD CV time: %.3f s' %np.std(t_time_test))
    print('')

    if analysis == 'vanilla_combi':
        # Train the entire dataset
        x_train_all, x_test_all, y_train_all, y_test_all = \
                train_test_split(x, y, test_size=.85, random_state=random_seed)
        print('All: Shape of the trainig and test dataset')
        print(y_train_all.shape, y_test_all.shape)
    elif analysis == 'uniform_dist':
        # Train the entire dataset
        x_train_all, x_test_all, y_train_all, y_test_all = \
                train_test_split(x, y, test_size=.20,  random_state=random_seed)
        print('ALL: Shape of the trainig and test dataset')
        print(y_train_all.shape, y_test_all.shape)
    print('Training RVR model:')
    model_all = RVR(kernel='linear')
    model_all.fit(x_train_all, y_train_all)
    # plot predicted vs true for the test (Entire sample)
    print('Plotting Predicted Vs True Age for all the sample')
    y_predicted_test = model.predict(x_test_all)
    output_path_test = save_path / ('rvr_test_predicted_true_age_rnd_seed%d.eps'
                                   %random_seed)
    plot_predicted_vs_true(y_test_all, y_predicted_test,
                           output_path_test, 'Age')

    return mae_cv, r_test, t_time_train, t_time_test


# Settings
# -----------------------------------------------------------------------------

# Number of cross validations
set_publication_style()
debug = False
dataset =  'freesurf_combined'
# dataset =  'UKBIO_freesurf'
bootstrap = 'bootstrap'
n_generations = 10 # generations on the TPOT
n_folds = 10 # number of times to perform Kfold
# Analysed random seeds
min_repetition = 10
max_repetition = 110
step_repetition = 10
save_path = Path('/code/BayOptPy/tpot_regression/Output/%s/age/%03d_generations' \
            %(args.analysis, n_generations))
save_path.mkdir(exist_ok=True, parents=True)

if bootstrap == 'bootstrap':
    print('Bootstrap Analysis')
    random_seeds = np.arange(min_repetition, max_repetition+step_repetition,
                             step_repetition)
    # iterate over the multiple random seeds
    mae_test_all = np.zeros((len(random_seeds), n_folds))
    r_test_all = np.zeros((len(random_seeds), n_folds))
    time_train_all = np.zeros((len(random_seeds), n_folds))
    time_test_all = np.zeros((len(random_seeds), n_folds))
    for seed_idx, random_seed in enumerate(random_seeds):
        mae_test, r_test, t_time_train, t_time_test  = rvr_analysis(random_seed, save_path, n_folds,
                                        args.analysis)
        mae_test_all[seed_idx, :] = mae_test.T
        r_test_all[seed_idx, :] = r_test.T
        time_train_all[seed_idx, :] = t_time_train
        time_test_all[seed_idx, :] = t_time_test
    print('Mean and std for test data')
    print(np.mean(mae_test_all, axis=0), np.std(mae_test_all, axis=0))
    print('Mean and std pearson corr test data')
    print(np.mean(r_test_all, axis=0), np.std(r_test_all, axis=0))
    print('Mean and std TIME training')
    print(np.mean(np.mean(time_train_all, axis=0)),
          np.std(np.std(time_train_all, axis=0)))
    print('Mean and std TIME test')
    print(np.mean(np.mean(time_test_all, axis=0)),
          np.std(np.std(time_test_all, axis=0)))

    results = {'mae_test': mae_test_all,
               'r_test': r_test_all,
               'time_train': time_train_all,
               'time_test': time_test_all}
    with open((save_path / 'rvr_all_seeds.pckl'), 'wb') as handle:
        pickle.dump(results, handle)
else:
    # Start timer
    print('Single Random Seed Analysis')
    t = time.process_time()

    # define default random seed
    random_seed = 20
    rvr_analysis(random_seed, save_path, n_folds, args.analysis)

    # Check elapsed time
    print('All script - Elapsed time in seconds:')
    elapsed_time = time.process_time() - t
    print('%.03f' %elapsed_time)
