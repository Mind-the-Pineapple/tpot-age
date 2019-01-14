import multiprocessing
import os
import argparse
from sklearn import model_selection
import numpy as np
#from dask.distributed import Client
import seaborn as sns
from matplotlib.pylab import plt
import pickle
from joblib import dump

from BayOptPy.tpot.extended_tpot import ExtendedTPOTRegressor
from BayOptPy.helperfunctions import get_data, get_paths, get_config_dictionary

parser = argparse.ArgumentParser()
parser.add_argument('-gui',
                    dest='gui',
                    action='store_true',
                    help='Use gui'
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
                    choices=['OASIS', 'BANC', 'BANC_freesurf']
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
parser.add_argument('-config_dict',
                    dest='config_dict',
                    help='Specify which TPOT config dict to use',
                    choices=['None', 'light', 'custom', 'ligth_no_preproc', 'gpr', 'gpr_full'],
                    required=True
                    )

args = parser.parse_args()

if __name__ == '__main__':

    print('The current args are: %s' %args)

    # check which TPOT dictionary containing the operators and parameters to be used was passed as argument
    if args.config_dict == 'None':
        tpot_config = None
    elif args.config_dict == 'light':
        tpot_config = 'TPOT light'
    elif args.config_dict == 'custom':
        from BayOptPy.tpot.custom_tpot_config_dict import tpot_config_custom
        tpot_config = tpot_config_custom
    elif args.config_dict == 'light_no_preproc':
        # this option uses the TPOT light definition without the preprocessing
        tpot_config = get_config_dictionary()
    elif args.config_dict == 'gpr':
        from BayOptPy.tpot.gpr_tpot_config_dict import tpot_config_gpr
        tpot_config = tpot_config_gpr
    elif args.config_dict == 'gpr_full':
        from BayOptPy.tpot.gpr_tpot_config_dict_full import tpot_config_gpr
        tpot_config = tpot_config_gpr

    random_seed = 42

    project_wd, project_data, project_sink = get_paths(args.debug, args.dataset)
    demographics, imgs, data = get_data(project_data, args.dataset, args.debug, project_wd, args.resamplefactor)

    print('Running regression analyis with TPOT')
    # split train-test dataset
    targetAttribute = np.array(demographics['Age'])
    if args.debug and args.dask:
        print('Start DASK client')
        port = 8889
    else:
        port = 8787
    if args.dask and args.debug:
        # TODO: These two ifs are not tested
        client = Client(threads_per_worker=1, diagnostics_port=port)
        client

    # To ensure the example runs quickly, we'll make the training dataset relatively small
    Xtrain, Xtest, Ytrain, Ytest = model_selection.train_test_split(data, targetAttribute, test_size=.25,
                                                                    random_state=random_seed)
    print('Divided dataset into test and training')
    print('Check train test split sizes')
    print('X_train: ' + str(Xtrain.shape))
    print('X_test: '  + str(Xtest.shape))
    print('Y_train: ' + str(Ytrain.shape))
    print('Y_test: '  + str(Ytest.shape))

    tpot = ExtendedTPOTRegressor(generations=args.generations,
                         population_size=args.population_size,
                         n_jobs=1,
                         cv=args.cv,
                         verbosity=2,
                         random_state=random_seed,
                         config_dict=tpot_config,
                         scoring='neg_mean_absolute_error',
                         use_dask=args.dask
                        )
    print('Number of cross-validation: %d' %args.cv)
    print('Number of generations: %d' %args.generations)
    print('Population Size: %d' %args.population_size)
    # njobs=-1 uses all cores present in the machine
    tpot.fit(Xtrain, Ytrain, Xtest)
    print('Test score using optimal model: %f ' % tpot.score(Xtest, Ytest))
    tpot.export(os.path.join(project_wd, 'BayOptPy', 'tpot', 'tpot_brain_age_pipeline.py'))
    print('Done TPOT analysis!')

    # Do some preprocessing to find models where all predictions have the same value and eliminate them, as those will correspond
    # to NaN entries or very small numbers on the correlation matrix.
    repeated_idx = np.argwhere(
        [np.array_equal(np.repeat(tpot.predictions[i][0], len(tpot.predictions[i])), tpot.predictions[i]) for i in
         range(len(tpot.predictions))])
    print('Index of the models with the same prediction for all subjects: ' + str(np.squeeze(repeated_idx)))
    print('Number of models analysed: %d' % len(tpot.predictions))
    tpot_predictions = np.delete(np.array(tpot.predictions), np.squeeze(repeated_idx), axis=0)

    print('Number of models that will be used for cross-correlation: %s' % (tpot_predictions.shape,))

    # Cross correlate the predictions
    corr_matrix = np.corrcoef(tpot_predictions)


    print('Check the number of NaNs after deleting models with constant predictions: %d' % len(
        np.argwhere(np.isnan(corr_matrix))))
    # colormap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr_matrix, cmap='coolwarm')
    plt.title(args.config_dict)
    plt.savefig(os.path.join(project_wd, 'BayOptPy', 'tpot', 'cross_corr_%s.png' % args.config_dict))

    # Dump tpot.pipelines and evaluated objects
    print('Dump correlation matrix and tpot pipelines')
    tpot_save = {}
    tpot_save['predictions'] = tpot.predictions
    tpot_save['evaluated_individuals_'] = tpot.evaluated_individuals_
    tpot_save['pipelines'] = tpot.pipelines
    tpot_save['corr_matrix'] = corr_matrix
    tpot_save['fitted_pipeline'] = tpot.fitted_pipeline_
    dump(tpot_save, os.path.join(project_wd, 'BayOptPy', 'tpot', 'tpot_%s_%s_%sgen_.dump') %(args.dataset, args.config_dict,
                                                                                             args.generations))

    if args.gui:
        plt.show()

