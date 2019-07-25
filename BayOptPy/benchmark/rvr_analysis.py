import os
import joblib
import pickle
import numpy as np

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_validate, cross_val_predict
from skrvm import RVR

from BayOptPy.helperfunctions import (get_paths, plot_predicted_vs_true,
                                      set_publication_style)

def rvr_analysis(random_seed, save_path):
    save_path = os.path.join(save_path, 'random_seed_%03d' %random_seed)
    print('Random seed: %03d' %random_seed)
    # Load the saved validation dataset
    project_ukbio_wd, project_data_ukbio, _ = get_paths(debug, dataset)
    with open(os.path.join(save_path, 'splitted_dataset_%s.pickle' %dataset), 'rb') as handle:
            splitted_dataset = pickle.load(handle)

    # Train the model
    model = RVR(kernel='linear')
    model.fit(splitted_dataset['Xtrain_scaled'], splitted_dataset['Ytrain'])

    scores = cross_validate(estimator= model,
                            X=splitted_dataset['Xtrain_scaled'],
                            y=splitted_dataset['Ytrain'],
                            scoring='neg_mean_absolute_error',
                            cv=n_cross_val)

    print("MAE train dataset: %0.2f (+/- %0.2f)" % (scores['test_score'].mean(),
                                                     scores['test_score'].std() * 2))

    # make cross validated predictions
    print('Perform prediction in test data')
    y_predicted_test = model.predict(splitted_dataset['Xtest_scaled'])
    output_path_test = os.path.join(save_path, 'test_predicted_true_age_rvr.png')
    plot_predicted_vs_true(splitted_dataset['Ytest'],  y_predicted_test,
                                                         output_path_test, 'Age')
    mae_test = mean_absolute_error(splitted_dataset['Ytest'],
                                         y_predicted_test)
    print('MAE on test: %.2f' %mae_test)

    print('Perform cross-validation in validation data')
    # y_predicted_validation = cross_val_predict(model,
    #                                 splitted_dataset['Xvalidate_scaled'],
    #                                 splitted_dataset['Yvalidate'],
    #                                 cv=n_cross_val)
    y_predicted_validation = model.predict(splitted_dataset['Xvalidate_scaled'])
    output_path_val = os.path.join(save_path, 'validation_predicted_true_age_rvr.png')
    plot_predicted_vs_true(splitted_dataset['Yvalidate'], y_predicted_validation,
                               output_path_val, 'Age')
    mae_validation = mean_absolute_error(splitted_dataset['Yvalidate'],
                                         y_predicted_validation)
    print('MAE on validation: %.2f' % mae_validation)

    # -----------------------------------------------------------------------------
    # Do some statistics. Calculate R2 and the Spearman
    from scipy.stats import spearmanr, pearsonr
    from sklearn.metrics import r2_score

    # Test dataset
    print('Statistics for the test dataset')
    rho_test, rho_p_value_test = spearmanr(splitted_dataset['Ytest'],
                                            y_predicted_test)
    print('shape of the dataset: %s' %(splitted_dataset['Ytest'].shape,))
    print('Rho and p-value: %.4f %.4f' %(rho_test, rho_p_value_test))
    r_test, r_p_value_test = pearsonr(splitted_dataset['Ytest'],
                                    y_predicted_test)
    print('R is: %.4f' %r_test)

    # Validation dataset
    print('Statistics for the validation dataset')
    rho_val, rho_p_value_val = spearmanr(splitted_dataset['Yvalidate'],
                                         y_predicted_validation)

    print('shape of the dataset: %s' %(splitted_dataset['Yvalidate'].shape,))
    print('Rho and p-value: %.4f %.4f' %(rho_val, rho_p_value_val))

    r_score = r2_score(splitted_dataset['Yvalidate'], y_predicted_validation)
    print('R2 is: %.4f' %r_score)

    r_val, r_p_value_val = pearsonr(splitted_dataset['Yvalidate'],
                                    y_predicted_validation)
    print('R is: %.4f' %r_val)
    print('-------------------------------------------------------------------')
    return mae_test, mae_validation, r_val, r_test


# Settings
# -----------------------------------------------------------------------------
# Number of cross validations
set_publication_style()
n_cross_val = 5
debug = False
dataset =  'freesurf_combined'
# dataset =  'UKBIO_freesurf'
analysis = 'bootstrap'
save_path = '/code/BayOptPy/tpot_regression/Output/vanilla_combi/age/100_generations/'


if analysis == 'bootstrap':
    random_seeds = np.arange(0, 110, 10)
    # iterate over the multiple random seeds
    mae_test_all = []
    mae_validation_all = []
    r_val_all = []
    r_test_all = []
    for random_seed in random_seeds:
        mae_test, mae_validation, r_val, r_test = rvr_analysis(random_seed, save_path)
        mae_test_all.append(mae_test)
        mae_validation_all.append(mae_validation)
        r_val_all.append(r_val)
        r_test_all.append(r_test)
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
    with open(os.path.join(save_path, 'rvr_all_seeds.pckl'), 'wb') as handle:
        pickle.dump(results, handle)
else:
    # define default random seed
    random_seed = 20
    rvr_analysis(random_seed)

