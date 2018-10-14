import os
import argparse
from tpot import TPOTRegressor
from sklearn import model_selection
import numpy as np
from dask.distributed import Client

from BayOptPy.helperfunctions import get_data

parser = argparse.ArgumentParser()
parser.add_argument('-nogui',
                    dest='nogui',
                    action='store_true',
                    help='No gui')
args = parser.parse_args()

if __name__ == '__main__':
    print('The current args are: %s' %args)

    #project_wd = '/BayOpt'
    project_wd = os.getcwd()
    project_sink, demographics, imgs, maskedData = get_data(project_wd)
    print('Running regression analyis with TPOT')
    # split train-test dataset
    targetAttribute = np.array(demographics['Age'])
    print('Start DASK client')
    client = Client(threads_per_worker=1)
    client

    Xtrain, Xtest, Ytrain, Ytest = model_selection.train_test_split(maskedData, targetAttribute, test_size=.4, random_state=42)
    print('Divided dataset into test and training')
    print('Check train test split sizes')
    print('X_train: ' + str(Xtrain.shape))
    print('X_test: '  + str(Xtest.shape))
    print('Y_train: ' + str(Ytrain.shape))
    print('Y_test: '  + str(Ytest.shape))

    tpot = TPOTRegressor(generations=5,
                         population_size=20,
                         n_jobs=-1,
                         verbosity=2,
                         # max_time_mins=20,
                         random_state=42,
                         config_dict='TPOT light',
                         use_dask=True
                         # memory='auto'
                        )
    # njobs=-1 uses all cores present in the machine
    tpot.fit(Xtrain, Ytrain)
    print(tpot.score(Xtest, Ytest))
    print('Done TPOT analysis!')
