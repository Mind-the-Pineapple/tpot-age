import os

import joblib
import pickle
import numpy as np

from BayOptPy.helperfunctions import (get_paths,
                                      set_publication_style,
                                      plot_predicted_vs_true)

'''
Plot predicted vs true values for the UKBIO bank cognitive metrics
'''

set_publication_style()
# General Settings
#-------------------------------------------------------------------------------
dataset = 'UKBIO_freesurf'
debug = False

# cog_tests = ['Fluid_intelligence', 'Reaction_time']
cog_tests = ['Reaction_time']

for cog_test in cog_tests:
    save_path ='/code/BayOptPy/tpot_regression/Output/vanilla_combi/%s/050_generations/random_seed_020' % cog_test

    tpot = joblib.load(os.path.join(save_path,
                                    'tpot_%s_vanilla_combi_050gen.dump'
                                   %dataset))
    exported_pipeline = tpot['fitted_pipeline']

    # Load the saved validation dataset
    project_ukbio_wd, project_data_ukbio, _ = get_paths(debug, dataset)
    with open(os.path.join(save_path, 'splitted_dataset_%s.pickle'
                           %dataset), 'rb') as handle:
                splitted_dataset = pickle.load(handle)

    y_predicted_test = exported_pipeline.predict(splitted_dataset['Xtest_scaled'])
    y_predicted_train = exported_pipeline.predict(splitted_dataset['Xtrain_scaled'])
    y_predicted_validation = exported_pipeline.predict(splitted_dataset['Xvalidate_scaled'])
    print('Predicted Value mean - test')
    print(np.mean(y_predicted_test))
    print('Predicted Value mean - train')
    print(np.mean(y_predicted_train))
    print('Predicted Value mean - Validation')
    print(np.mean(y_predicted_validation))

    # plot predicted vs true for the test
    output_path_test = os.path.join(save_path, 'test_predicted_true.png')
    plot_predicted_vs_true(splitted_dataset['Ytest'], y_predicted_test,
    output_path_test, cog_test)

    # plot predicted vs true for the validation
    output_path_val = os.path.join(save_path, 'validation_predicted_true.png')
    plot_predicted_vs_true(splitted_dataset['Yvalidate'], y_predicted_validation,
    output_path_val, cog_test)

