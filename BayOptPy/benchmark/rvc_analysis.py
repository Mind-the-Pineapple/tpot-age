import os
import joblib
import pickle

import numpy as np
from matplotlib.pylab import plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate, cross_val_predict
from skrvm import RVC

from BayOptPy.helperfunctions import (get_paths,
                                      set_publication_style,
                                      plot_confusion_matrix)

def rvc_analysis(random_seed, save_path):
    # Load the data
    # TODO: change the path
    save_path = os.path.join(save_path, 'random_seed_%03d' %random_seed)
    print('Random seed: %03d' %random_seed)
    # Load the saved validation dataset
    project_ukbio_wd, project_data_ukbio, _ = get_paths(debug, dataset)
    with open(os.path.join(save_path, 'splitted_dataset_%s.pickle' %dataset), 'rb') as handle:
            splitted_dataset = pickle.load(handle)

    # Train the model
    model = RVC(kernel='linear')
    model.fit(splitted_dataset['Xtrain_scaled'], splitted_dataset['Ytrain'])

    # make cross validated predictions
    print('Perform prediction in test data')
    y_prediction_test = model.predict(splitted_dataset['Xtest_scaled'])

    y_prediction_validation = model.predict(splitted_dataset['Xvalidate_scaled'])

    # -----------------------------------------------------------------------------
    # Do some statistics. Calculate the confusion matrix

    # Test dataset
    # Look at the confusion matrix for test data
    class_name = np.array(['young', 'old', 'adult'], dtype='U10')
    ax, cm_test = plot_confusion_matrix(splitted_dataset['Ytest'], y_prediction_test,
                          classes=class_name,
                          normalize=True)
    # Look at accuracy
    accuracy_test = accuracy_score(splitted_dataset['Ytest'], y_prediction_test)
    plt.savefig(os.path.join(save_path, 'confusion_matrix_test_rvc.eps'))

   # Predict on the validation dataset
    ax, cm_validation = plot_confusion_matrix(splitted_dataset['Yvalidate'], y_prediction_validation,
                          classes=class_name,
                          normalize=True)
    plt.savefig(os.path.join(save_path, 'confusion_matrix_validation_rvc.eps'))
    # Look at accuracy
    accuracy_val = accuracy_score(splitted_dataset['Yvalidate'],
                                   y_prediction_validation)
    plt.savefig(os.path.join(save_path, 'confusion_matrix_test_rvc.eps'))
    return cm_test, cm_validation, accuracy_test, accuracy_val


# Settings
# -----------------------------------------------------------------------------
# Number of cross validations
set_publication_style()
n_cross_val = 5
debug = False
dataset =  'freesurf_combined'
# dataset =  'UKBIO_freesurf'
analysis = 'bootstrap'
n_generations = 5
save_path = '/code/BayOptPy/tpot_classification/Output/vanilla_combi/age/%03d_generations/' %(n_generations)

if analysis =='bootstrap' :
    random_seeds = np.arange(0, 100, 10)
    # iterate over the multiple random seeds
    confusion_matrix_test_all = []
    confusion_matrix_validate_all = []
    accuracy_test_all = []
    accuracy_val_all = []
    for random_seed in random_seeds:
        cm_test, cm_validation, accuracy_test, accuracy_val  = rvc_analysis(random_seed, save_path)
        confusion_matrix_test_all.append(cm_test)
        confusion_matrix_validate_all.append(cm_validation)
        accuracy_test_all.append(accuracy_test)
        accuracy_val_all.append(accuracy_val)

print('Mean and std for test data')
print(np.mean(confusion_matrix_test_all, axis=0),
      np.std(confusion_matrix_test_all, axis=0))
print('Mean and std for validation data')
print(np.mean(confusion_matrix_validate_all, axis=0),
      np.std(confusion_matrix_validate_all, axis=0))

results = {'confusion_matrix_test': confusion_matrix_test_all,
           'confusion_matrix_validation': confusion_matrix_validate_all,
           'score_test': accuracy_test_all,
           'score_val': accuracy_val_all,
          }
with open(os.path.join(save_path, 'rvc_all_seeds.pckl'), 'wb') as handle:
    pickle.dump(results, handle)
