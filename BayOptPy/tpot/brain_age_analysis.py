import multiprocessing
import os
import argparse
from sklearn import model_selection
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.pylab import plt
import seaborn as sns
import pickle
# from sklearn.externals import joblib
import joblib


from BayOptPy.tpot.extended_tpot import ExtendedTPOTRegressor
from BayOptPy.helperfunctions import (get_data, get_paths,
                                      get_config_dictionary, get_output_path,
                                      get_best_pipeline_paths)

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
                    help="Specify population size to use. This value specifiy \
                    the number of individuals to retain in the genetic         \
                    programiming population at every generation.",
                    type=int,
                    default=100 # use the same default as TPOT default population value
                    )
parser.add_argument('-offspring_size',
                    dest='offspring_size',
                    help='Specify offspring size to use. This value corresponds\
                    to the number of offsprings to produce in each genetic     \
                    programming generation. By default, the number of          \
                    offsprings is equal to the number of the population size.',
                    type=int,
                    default=None # When None is passed use the same as
                                 # population_size (TPOT default)
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
                    choices=['None', 'light', 'custom',
                             'gpr', 'gpr_full', 'vanilla', 'feat_selec',
                             'feat_combi', 'vanilla_combi'],
                    required=True
                    )
parser.add_argument('-njobs',
                     dest='njobs',
                     type=int,
                     required=True)
parser.add_argument('-random_seed',
                    dest='random_seed',
                    help='Specify random seed to use',
                    type=int,
                    required=True
                    )
parser.add_argument('-analysis',
                    dest='analysis',
                    help='Specify which type of analysis to use',
                    choices=['vanilla', 'population', 'feat_selec',
                             'feat_combi', 'vanilla_combi', 'mutation',
                             'random_seed'],
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


    print('The current args are: %s' %args)

    # check which TPOT dictionary containing the operators and parameters to be used was passed as argument
    if args.config_dict == 'None':
        tpot_config = None
    elif args.config_dict == 'light':
        tpot_config = 'TPOT light'
    elif args.config_dict == 'custom':
        from BayOptPy.tpot.custom_tpot_config_dict import tpot_config_custom
        tpot_config = tpot_config_custom
    elif args.config_dict == 'gpr':
        from BayOptPy.tpot.gpr_tpot_config_dict import tpot_config_gpr
        tpot_config = tpot_config_gpr
    elif args.config_dict == 'gpr_full':
        from BayOptPy.tpot.gpr_tpot_config_dict_full import tpot_config_gpr
        tpot_config = tpot_config_gpr
    elif args.config_dict == 'vanilla':
        from BayOptPy.tpot.gpr_tpot_config_vanilla import tpot_config_gpr
        tpot_config = tpot_config_gpr
    elif args.config_dict == 'feat_selec':
        #Load models for feature selection
        from BayOptPy.tpot.gpr_tpot_config_feat_selec import tpot_config_gpr
        tpot_config = tpot_config_gpr
    elif args.config_dict == 'feat_combi':
        # Load models for feature combination
        from BayOptPy.tpot.gpr_tpot_config_feat_combi import tpot_config_gpr
        tpot_config = tpot_config_gpr
    elif args.config_dict == 'vanilla_combi':
        # Load models for feature combination
        from BayOptPy.tpot.gpr_tpot_config_feat_combi import tpot_config_gpr
        tpot_config = tpot_config_gpr

    # Get data paths, the actual data and check if the output paths exists
    project_wd, project_data, project_sink = get_paths(args.debug, args.dataset)
    output_path = get_output_path(args.analysis, args.generations, args.random_seed,
                                  args.population_size, args.debug,
                                  args.mutation_rate, args.crossover_rate)
    demographics, imgs, data, _ = get_data(project_data, args.dataset, args.debug, project_wd, args.resamplefactor)

    # Show mean std and F/M count for each dataset used
    aggregations = {
        'Age': ['mean', 'std', 'min', 'max'],
        'sex': 'size'
    }
    demographics.groupby(['original_dataset', 'sex']).aggregate(aggregations)

    # Print total N per dataset
    for dataset in np.unique(demographics['original_dataset']):
        print('%s: %d' % (dataset, sum(demographics['original_dataset'] == dataset)))

    # Print demographics for the final dataset
    demographics.groupby(['sex']).aggregate(aggregations)

    # Path to the folder where to save the best pipeline will be saved
    # Note: The pipeline will only be saved if it is different from the one in
    # the previous generation
    best_pipe_paths = get_best_pipeline_paths(args.analysis, args.generations,
                                              args.random_seed,
                                              args.population_size,
                                              args.debug,
                                              args.mutation_rate,
                                              args.crossover_rate)
    print('Checkpoint folder path')
    print(best_pipe_paths)

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

    Xtrain, Xtest, Ytrain, Ytest = model_selection.train_test_split(data, targetAttribute, test_size=.25,
                                                                    random_state=args.random_seed)
    print('Divided dataset into test and training')
    print('Check train test split sizes')
    print('X_train: ' + str(Xtrain.shape))
    print('X_test: '  + str(Xtest.shape))
    print('Y_train: ' + str(Ytrain.shape))
    print('Y_test: '  + str(Ytest.shape))

    tpot = ExtendedTPOTRegressor(generations=args.generations,
                         population_size=args.population_size,
                         offspring_size=args.offspring_size,
                         mutation_rate=args.mutation_rate,
                         crossover_rate=args.crossover_rate,
                         n_jobs=args.njobs,
                         cv=args.cv,
                         verbosity=2,
                         random_state=args.random_seed,
                         config_dict=tpot_config,
                         scoring='neg_mean_absolute_error',
                         periodic_checkpoint_folder=best_pipe_paths,
                         use_dask=args.dask,
                         debug=args.debug,
                         analysis=args.analysis,
                        )
    print('Number of cross-validation: %d' %args.cv)
    print('Number of generations: %d' %args.generations)
    print('Population Size: %d' %args.population_size)
    print('Offspring Size: %d' %args.offspring_size)
    # njobs=-1 uses all cores present in the machine
    tpot.fit(Xtrain, Ytrain, Xtest, Ytest)
    print('Test score using optimal model: %f ' % tpot.score(Xtest, Ytest))
    tpot.export(os.path.join(project_wd, 'BayOptPy', 'tpot', 'tpot_brain_age_pipeline.py'))
    print('Done TPOT analysis!')

    print('Number of models analysed: %d' % len(tpot.predictions))
    repeated_idx = np.argwhere(
          [np.array_equal(np.repeat(tpot.predictions[i][0], len(tpot.predictions[i])),
                          tpot.predictions[i]) for i in range(len(tpot.predictions))])
    print('Index of the models with the same prediction for all subjects: ' + str(np.squeeze(repeated_idx)))
    tpot_predictions = np.delete(np.array(tpot.predictions), np.squeeze(repeated_idx), axis=0)

    # Dump tpot.pipelines and evaluated objects
    print('Dump predictions, evaluated pipelines and sklearn objects')
    tpot_save = {}
    tpot_pipelines = {}
    tpot_save['Xtest'] = Xtest
    tpot_save['Ytest'] = Ytest
    tpot_save['predictions'] = tpot.predictions
    # List of
    tpot_save['evaluated_individuals_'] = tpot.evaluated_individuals_
    # Best pipeline at the end of the genetic algorithm
    tpot_save['fitted_pipeline'] = tpot.fitted_pipeline_
    # List of evaluated invidivuals per generation
    tpot_save['evaluated_individuals'] = tpot.evaluated_individuals
    # List of best model per generation
    tpot_pipelines['log'] = tpot.log

    # Dump results
    joblib.dump(tpot_save, os.path.join(output_path, 'tpot_%s_%s_%03dgen.dump')
                                 %(args.dataset, args.config_dict,
                                   args.generations))
    joblib.dump(tpot_pipelines, os.path.join(output_path,
                                      'tpot_%s_%s_%03dgen_pipelines.dump')
                                      %(args.dataset, args.config_dict,
                                        args.generations))

    if args.gui:
        plt.show()

