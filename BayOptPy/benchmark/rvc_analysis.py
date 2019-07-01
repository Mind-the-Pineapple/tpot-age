import os
import joblib
import pickle

import numpy as np
from matplotlib.pylab import plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_validate, cross_val_predict
from skrvm import RVC

from BayOptPy.helperfunctions import (get_paths, plot_predicted_vs_true_age,
                                      set_publication_style,
                                      plot_confusion_matrix)

# Settings
# -----------------------------------------------------------------------------
# Number of cross validations
set_publication_style()
n_cross_val = 5
debug = False
dataset =  'freesurf_combined'
# dataset =  'UKBIO_freesurf'


# Load the data
# TODO: change the path
save_path = '/code/BayOptPy/tpot_classification/Output/vanilla_combi/005_generations/random_seed_020/'
# Load the saved validation dataset
project_ukbio_wd, project_data_ukbio, _ = get_paths(debug, dataset)
with open(os.path.join(save_path, 'splitted_dataset_%s.pickle' %dataset), 'rb') as handle:
        splitted_dataset = pickle.load(handle)
# Train the model
model = RVC(kernel='linear')
model.fit(splitted_dataset['Xtrain_scaled'], splitted_dataset['Ytrain'])

scores = cross_validate(estimator= model,
                        X=splitted_dataset['Xtrain_scaled'],
                        y=splitted_dataset['Ytrain'],
                        scoring='neg_mean_absolute_error',
                        cv=n_cross_val)


# make cross validated predictions
print('Perform prediction in test data')
y_prediction_test = model.predict(splitted_dataset['Xtest_scaled'])

y_prediction_validation = model.predict(splitted_dataset['Xvalidate_scaled'])

# -----------------------------------------------------------------------------
# Do some statistics. Calculate R2 and the Spearman

# Test dataset
# Look at the confusion matrix for test data
class_name = np.array(['young', 'old', 'adult'], dtype='U10')
plot_confusion_matrix(splitted_dataset['Ytest'], y_prediction_test,
                      classes=class_name,
                      normalize=True)
plt.savefig(os.path.join(save_path, 'confusion_matrix_test_rvc.png'))
# Predict on the validation dataset
import pdb
pdb.set_trace()
plot_confusion_matrix(splitted_dataset['Yvalidate'], y_prediction_validation,
                      classes=class_name,
                      normalize=True)
plt.savefig(os.path.join(save_path, 'confusion_matrix_validation_rvc.png'))
