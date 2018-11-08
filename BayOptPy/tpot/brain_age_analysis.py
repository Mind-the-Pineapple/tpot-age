import multiprocessing
import os
import argparse
from tpot import TPOTRegressor
from sklearn import model_selection
import numpy as np
import pickle
#from dask.distributed import Client

from BayOptPy.helperfunctions import get_data, get_paths, get_config_dictionary

parser = argparse.ArgumentParser()
parser.add_argument('-nogui',
                    dest='nogui',
                    action='store_true',
                    help='No gui'
                    )
parser.add_argument('-debug',
                    dest='debug',
                    action='store_true',
                    help='Run debug with Pycharm'
                    )
parser.add_argument('-dask',
                    dest='dask',
                    action='store_true',
                    help='Run analysis with dask'
                    )
parser.add_argument('-dataset',
                    dest='dataset',
                    help='Specify which dataset to use',
                    choices=['OASIS', 'BANC']
                    )
parser.add_argument('-cv',
                    dest='cv',
                    help='Specify number of cross validations to use',
                    type=int,
                    required=True
                    )
parser.add_argument('-generations',
                    dest='generations',
                    help='Specify number of generations to use',
                    type=int,
                    required=True
                    )
parser.add_argument('-population_size',
                    dest='population_size',
                    help='Specify population size to use',
                    type=int,
                    default=100 # use the same default as TPOT default population value
                    )
parser.add_argument('-resamplefactor',
                    dest='resamplefactor',
                    help='Specify resampling rate for the image affine',
                    type=int,
                    default=1 # no resampling is performed
                    )
parser.add_argument('-nopreprocessing',
                    dest='nopreprocessing',
                    action='store_true',
                    help='Use default TPOT light models without the preprocessing',
                    default=False
                    )

args = parser.parse_args()

if __name__ == '__main__':

    print('The current args are: %s' %args)

    project_wd, project_data, project_sink = get_paths(args.debug, args.dataset)
    demographics, imgs, maskedData = get_data(project_data, args.dataset, args.debug, project_wd, args.resamplefactor)

    print('Running regression analyis with TPOT')
    # split train-test dataset
    targetAttribute = np.array(demographics['Age'])
    if args.debug and args.dask:
        print('Start DASK client')
        port = 8889
    else:
        port = 8787
    if args.dask:
        client = Client(threads_per_worker=1, diagnostics_port=port)
        client

    # To ensure the example runs quickly, we'll make the training dataset relatively small
    Xtrain, Xtest, Ytrain, Ytest = model_selection.train_test_split(maskedData, targetAttribute, test_size=.5, random_state=42)
    print('Divided dataset into test and training')
    print('Check train test split sizes')
    print('X_train: ' + str(Xtrain.shape))
    print('X_test: '  + str(Xtest.shape))
    print('Y_train: ' + str(Ytrain.shape))
    print('Y_test: '  + str(Ytest.shape))

    # if no preprocessing is passed as argument do not perform any preprocessing on the pipelines
    if args.nopreprocessing:
        config_dic = get_config_dictionary()
    else:
        config_dic = 'TPOT_light'

    tpot = TPOTRegressor(generations=args.generations,
                         population_size=args.population_size,
                         n_jobs=1,
                         cv=args.cv,
                         verbosity=2,
                         # max_time_mins=20,
                         random_state=42,
                         config_dict=config_dic,
                         scoring='neg_mean_absolute_error',
                         use_dask=args.dask
                         # memory='auto'
                        )
    print('Number of cross-validation: %d' %args.cv)
    print('Number of generations: %d' %args.generations)
    print('Population Size: %d' %args.population_size)
    # njobs=-1 uses all cores present in the machine
    tpot.fit(Xtrain, Ytrain)
    print(tpot.score(Xtest, Ytest))
    tpot.export('tpot_simple_analysis_pipeline.py')
    print('Done TPOT analysis!')

    # Pickle dictionary with the the list of evaluated pipelines
    with open(os.path.join(project_wd, 'BayOptPy', 'tpot', '%s_pipelines.pkl' %args.dataset), 'wb') as handle:
        pickle.dump(tpot.evaluated_individuals_, handle)
