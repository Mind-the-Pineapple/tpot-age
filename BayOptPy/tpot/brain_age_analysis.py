import multiprocessing
import os
import argparse
import pickle
import time
from pathlib import Path

from sklearn import model_selection
from sklearn.preprocessing import RobustScaler
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.pylab import plt
import seaborn as sns
import joblib
import pandas as pd
from tpot import TPOTClassifier

from BayOptPy.tpot.extended_tpot import ExtendedTPOTRegressor
from BayOptPy.helperfunctions import (get_data, get_paths,
                                      get_config_dictionary, get_output_path,
                                      get_best_pipeline_paths,
                                      create_age_histogram,
                                      plot_confusion_matrix,
                                      load_cognitive_data,
                                      get_uniform_dist_data)

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
parser.add_argument('-raw',
                    dest='raw',
                    help='Use raw freesurfer dataset',
                    choices=['True', 'False'],
                    required=True
                    )
parser.add_argument('-model',
                    dest='model',
                    help='Define if a classification or regression problem',
                    choices=['regression', 'classification', 'classification2']
                    )
parser.add_argument('-predicted_attribute',
                   dest='predicted_attribute',
                   help='Define the cognitive task of interest',
                   choices=['age', 'Reaction_time', 'Prospective_memory',
                    'Fluid_intelligence', 'gender']
                   )
parser.add_argument('-dask',
                    dest='dask',
                    action='store_true',
                    help='Run analysis with dask'
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
                             'feat_combi', 'vanilla_combi',
                             'vanilla_classification'],
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
                             'random_seed', 'ukbio', 'summary_data',
                             'uniform_dist'],
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

def str_to_bool(s):
    '''
    As arg pass does not acess boolen, transfrom the string into booleans
    '''
    if s == 'True':
        return True
    elif s == 'False':
        return False

args = parser.parse_args()

if __name__ == '__main__':


    print('-----------------------------------------------------------------')
    print('Perform arguments check:')
    print('-----------------------------------------------------------------')
    # A few test to check the input arguments
    if args.predicted_attribute == 'Prospective_memory' and args.model != 'classification':
        parser.error('Prospective_memory requires a classifer')
    if args.predicted_attribute == 'Fluid_intelligence' and args.model != 'regression':
        parser.error('Prospective_memory requires a regression')
    if args.predicted_attribute == 'Reaction_time' and args.model != 'regression':
        parser.error('Prospective_memory requires a regression')
    if args.predicted_attribute != 'age' and args.dataset != 'UKBIO_freesurf':
        parser.error('Cognitive and gender analysis is only implemented for \
                     the UKBIO bank dataset')
    if args.model == 'classification' and args.config_dict not in \
                                             ['vanilla_classification', 'None']:
        parser.error('The only dictionary implemented for classification \
                     analysis is the vanilla_classification and None')
    if args.model == 'classification2' and args.config_dict not in \
                                             ['vanilla_classification', 'None']:
        parser.error('The only dictionary implemented for classification \
                     analysis is the vanilla_classification and None')
    if args.analysis == 'uniform_dist' and args.model != 'regression':
        parser.error('Uniform distribution only tested for regression problemsr')
    print('The current args are: %s' %args)
    print(' ')

    print('-----------------------------------------------------------------')
    print('Load dictionary of models to use:')
    print('-----------------------------------------------------------------')
    # check which TPOT dictionary containing the operators and parameters to be
    # used was passed as argument. The dictionaries are different depeding if
    # the analysis is a classification or a regression problem
    if args.model == 'classification' or args.model == 'classification2' :
        if args.config_dict == 'None':
            tpot_config = None
        elif args.config_dict == 'vanilla_classification':
            from BayOptPy.tpot.dicts.tpot_config_vanilla_classification import tpot_config_gpr
            tpot_config = tpot_config_gpr
    if args.model == 'regression':
        if args.config_dict == 'None':
            tpot_config = None
        elif args.config_dict == 'light':
            tpot_config = 'TPOT light'
        elif args.config_dict == 'custom':
            from BayOptPy.tpot.dicts.custom_tpot_config_dict import tpot_config_custom
            tpot_config = tpot_config_custom
        elif args.config_dict == 'gpr':
            from BayOptPy.tpot.dicts.gpr_tpot_config_dict import tpot_config_gpr
            tpot_config = tpot_config_gpr
        elif args.config_dict == 'gpr_full':
            from BayOptPy.tpot.dicts.gpr_tpot_config_dict_full import tpot_config_gpr
            tpot_config = tpot_config_gpr
        elif args.config_dict == 'vanilla':
            from BayOptPy.tpot.dicts.gpr_tpot_config_vanilla import tpot_config_gpr
            tpot_config = tpot_config_gpr
        elif args.config_dict == 'feat_selec':
            #Load models for feature selection
            from BayOptPy.tpot.dicts.gpr_tpot_config_feat_selec import tpot_config_gpr
            tpot_config = tpot_config_gpr
        elif args.config_dict == 'feat_combi':
            # Load models for feature combination
            from BayOptPy.tpot.dicts.gpr_tpot_config_feat_combi import tpot_config_gpr
            tpot_config = tpot_config_gpr
        elif args.config_dict == 'vanilla_combi':
            # Load models for feature combination
            from BayOptPy.tpot.dicts.gpr_tpot_config_vanilla_combi import tpot_config_gpr
            tpot_config = tpot_config_gpr


    print('-----------------------------------------------------------------')
    print('Get datapaths:')
    print('-----------------------------------------------------------------')
    # Get data paths, the actual data and check if the output paths exists
    project_wd, project_data, project_sink = get_paths(args.debug, args.dataset)
    output_path = get_output_path(args.model, args.analysis, args.generations,
                                  args.random_seed,
                                  args.population_size, args.debug,
                                  args.mutation_rate, args.crossover_rate,
                                  args.predicted_attribute)
    # Load the already cleaned dataset
    demographics, imgs, dataframe  = get_data(project_data, args.dataset,
                                              args.debug, project_wd,
                                              args.resamplefactor,
                                              raw=str_to_bool(args.raw),
                                              analysis=args.analysis)
    print('Using %d features' %len(dataframe.columns))

    # If we are looking at the uniform distribution, get the corresponding
    # dataset
    if args.analysis == 'uniform_dist':
        demographics, dataframe = get_uniform_dist_data(args.debug, args.dataset,
                                                        args.resamplefactor,
                                                        str_to_bool(args.raw),
                                                        args.analysis)

    if args.dataset == 'freesurf_combined' or args.dataset == 'UKBIO_freesurf':
        # Drop the last coumn which correspond to the dataset name
        dataframe = dataframe.drop(['dataset'], axis=1)

    # Select the features related only to cortical volume and thickness (only
    # for the UKBIO)
    if args.dataset == 'UKBIO_freesurf' and args.raw =='True':
        from freesurfer_columns import thk_and_vol as COLUMN_NAMES

    # Show mean std and F/M count for each dataset used
    aggregations = {
        'age': ['mean', 'std', 'min', 'max'],
        'sex': 'size'
    }
    if args.dataset == 'BANC_freesurf':
        demographics.groupby(['original_dataset', 'sex']).aggregate(aggregations)

        # Print total N per dataset
        for dataset in np.unique(demographics['original_dataset']):
            print('%s: %d' % (dataset, sum(demographics['original_dataset'] == dataset)))


    # Print demographics for the final dataset
    demographics.groupby(['sex']).aggregate(aggregations)
    demographics = demographics.set_index('id')
    # Path to the folder where to save the best pipeline will be saved
    # Note: The pipeline will only be saved if it is different from the one in
    # the previous generation
    best_pipe_paths = get_best_pipeline_paths(args.model,
                                              args.analysis, args.generations,
                                              args.random_seed,
                                              args.population_size,
                                              args.debug,
                                              args.mutation_rate,
                                              args.crossover_rate,
                                              args.predicted_attribute)
    print('Checkpoint folder path')
    print(best_pipe_paths)

    # Plot age distribution for the training and test dataset
    create_age_histogram(demographics, args.dataset)

    print(' ')
    print('-----------------------------------------------------------------')
    print('Get target Attribute:')
    print('-----------------------------------------------------------------')
    print(args.predicted_attribute)
    # Select target attribute
    if args.predicted_attribute == 'age':
        targetAttribute = demographics['age']
    elif args.predicted_attribute == 'gender':
        # transform female/male into numeric values
        gender = {'male':0, 'female':1}
        demographics['gender'] = [gender[sub] for sub in demographics['sex']]
        targetAttribute = demographics['gender']

    else:
        # Load the cognitive test data
        cog_df = load_cognitive_data(project_data)
        # Select demographics only for the subjects that we have cognitive
        # information
        demographics = demographics[demographics.index.isin(cog_df.index)]
        # add age to the list of cog_df values
        cog_df['age'] = demographics['age']
        # Plot pairplot between reaction all features in the dataframe
        plt.figure()
        ax = sns.pairplot(cog_df)
        plt.savefig(os.path.join(project_data, 'cog_ukbio', 'pairplot.eps'))
        plt.close()

        # Split the data into train, test, validate
        targetAttribute = cog_df[args.predicted_attribute].dropna()
        # Remove the subjects who has NaN as values for this targetAttribute
        dataframe = dataframe.loc[targetAttribute.index]

        if args.predicted_attribute == 'Prospective_memory':
            # Because there is an imbalanced distribution. Way more subjects are
            # classified as First Attempt then other categories. We trim the
            # sample size down so that all 3 categories are evenly distributed.

            # randomly select 190 (lower bound) for all 3 categories
            selected_idx = []
            for category in targetAttribute.value_counts().index:
                selected_idx.append(np.random.choice(
                            targetAttribute[targetAttribute == category].index.values, 190,
                                           replace=False))
            flatten_selected_idx = [x for sublist in selected_idx for x in
                                    sublist]
            dataframe = dataframe[dataframe.index.isin(flatten_selected_idx)]
            targetAttribute = targetAttribute[targetAttribute.index.isin(flatten_selected_idx)]
    if args.debug and args.dask:
        print('Start DASK client')
        port = 8889
    else:
        port = 8787
    if args.dask and args.debug:
        # TODO: These two ifs are not tested
        client = Client(threads_per_worker=1, diagnostics_port=port)
        client

    print(' ')
    print('-----------------------------------------------------------------')
    print('Split train, test, validation dataset:')
    print('-----------------------------------------------------------------')
    if args.dataset == 'freesurf_combined':
    # This assumes that your dataframe already has a column that defines the
    # popularity of every group M/F and age in the dataset
        def classification_groups(col):
            if col <= 30:
                return 0
            if (col> 31) and (col<=69):
                return 1
            if col>70:
                return 2

        if args.model == 'classification' and args.predicted_attribute == 'age':
            # Transform the age into classes
            demographics['classification_groups'] = demographics['age'].apply(classification_groups)

            young_demographics = demographics.loc[demographics['age']<= 30]
            adults_demographics = demographics.loc[(demographics['age']> 31) &
                                                   (demographics['age']<= 69)]
            old_demographics = demographics.loc[demographics['age'] > 70]
            # get the demographics for each new gruop
            demographics = pd.concat([young_demographics[:600],
                                      adults_demographics[:600],
                                      old_demographics[:600]])
            # Select the data for the corresponding subjects
            dataframe = dataframe.loc[demographics.index]
            targetAttribute = demographics['classification_groups']

            # Classification will separate the training, test and validation
            # dataset using stratification
            Xtrain, Xtemp, Ytrain, Ytemp = \
                model_selection.train_test_split(dataframe, targetAttribute,
                                                 test_size=.65,
                                                 random_state=args.random_seed)
            # Get the stratified list for the training dataset
            train_demographics = demographics.loc[Xtemp.index]
            Xvalidate, Xtest, Yvalidate, Ytest = \
                    model_selection.train_test_split(Xtemp, Ytemp,
                                                     test_size=.5,
                                                     random_state=args.random_seed)



        elif args.model == 'classification2' and args.predicted_attribute == 'age':
            # Use only 500 young subjects (aged 18-30) and 500 old subjects
            # (70-90).
            young_demographics = demographics.loc[demographics['age'] < 30]
            old_demographics = demographics.loc[demographics['age'] > 70]
            demographics = pd.concat([young_demographics[:600],
                                     old_demographics[:600]])
            print('new demographics shape: %s' %(demographics.shape,))
            # Select the data for the corresponding subjects
            dataframe = dataframe.loc[demographics.index]
            targetAttribute = targetAttribute.loc[demographics.index]
            # Split tain, test and validate
            Xtrain, Xtemp, Ytrain, Ytemp = \
                model_selection.train_test_split(dataframe, targetAttribute,
                                                 test_size=.5,
                                                 random_state=args.random_seed)
            # Get the stratified list for the training dataset
            train_demographics = demographics.loc[Xtemp.index]
            Xvalidate, Xtest, Yvalidate, Ytest = \
                    model_selection.train_test_split(Xtemp, Ytemp,
                                                     test_size=.5,
                                                     random_state=args.random_seed)

        elif (args.model == 'regression') and (args.analysis!= 'uniform_dist'):
            # Split tain, test and validate
            Xtrain, Xtest, Ytrain, Ytest = \
                model_selection.train_test_split(dataframe, targetAttribute,
                                                 test_size=.85,
                                                 random_state=args.random_seed)

        elif (args.model == 'regression') and (args.analysis == 'uniform_dist'):
            # Split tain, test and validate
            Xtrain, Xtest, Ytrain, Ytest = \
                model_selection.train_test_split(dataframe, targetAttribute,
                                                 test_size=.20,
                                                 random_state=args.random_seed)

    elif args.dataset == 'UKBIO_freesurf':
            if args.predicted_attribute == 'Prospective_memory':
                # Split tain, test and validate
                Xtrain, Xtemp, Ytrain, Ytemp = \
                    model_selection.train_test_split(dataframe, targetAttribute,
                                                     test_size=.5,
                                                     random_state=args.random_seed)
                # Get the stratified list for the training dataset
                train_demographics = demographics.loc[Xtemp.index]
                Xvalidate, Xtest, Yvalidate, Ytest = \
                        model_selection.train_test_split(Xtemp, Ytemp,
                                                         test_size=.5,
                                                         random_state=args.random_seed)

            else:
                # Split tain, test and validate
                Xtrain, Xtemp, Ytrain, Ytemp = \
                    model_selection.train_test_split(dataframe, targetAttribute,
                                                     test_size=.85,
                                                     random_state=args.random_seed)
                # Get the stratified list for the training dataset
                train_demographics = demographics.loc[Xtemp.index]
                Xvalidate, Xtest, Yvalidate, Ytest = \
                        model_selection.train_test_split(Xtemp, Ytemp,
                                                         test_size=.5,
                                                         random_state=args.random_seed)

    else:
        Xtrain, Xtest, Ytrain, Ytest = \
                model_selection.train_test_split(dataframe, targetAttribute,
                                                 test_size=.25,
                                                 random_state=args.random_seed)


    # Check the group distribution
    # It is not that easy because your labels are not thesabe as Y. But the mean
    # age of the Ytrain and Ytest is vvery similar!
    # print(np.unique(Ytrain, return_counts=True))
    # print(np.unique(Ytest, return_counts=True))
    print('Divided dataset into test and training')
    print('Check train test split sizes')
    print('X_train: ' + str(Xtrain.shape))
    print('X_test: '  + str(Xtest.shape))
    print('Y_train: ' + str(Ytrain.shape))
    print('Y_test: '  + str(Ytest.shape))

    # Normalise the test dataset and apply the transformation to the train
    # dataset
    robustscaler = RobustScaler().fit(Xtrain)
    Xtrain_scaled = robustscaler.transform(Xtrain)
    Xtest_scaled = robustscaler.transform(Xtest)

    if args.dataset == 'freesurf_combined' or args.dataset == 'UKBIO_freesurf':
        # print('Y_validate: ' + str(Yvalidate.shape))
        # print('X_validate: ' + str(Xvalidate.shape))

        # Xvalidate_scaled = robustscaler.transform(Xvalidate)
        # Dump the validation set and delete the loaded subjects
        validation = {
                      # 'Xvalidate': Xvalidate,
                      # 'Yvalidate': Yvalidate,
                      # 'Xvalidate_scaled': Xvalidate_scaled,
                      'Xtrain': Xtrain,
                      'Ytrain': Ytrain,
                      'Xtrain_scaled': Xtrain_scaled,
                      'Xtest': Xtest,
                      'Ytest': Ytest,
                      'Xtest_scaled': Xtest_scaled}
        with open(os.path.join(output_path, 'splitted_dataset_%s.pickle'
                               %args.dataset), 'wb') as handle:
            pickle.dump(validation, handle)

    # Create classes for the classification
    #--------------------------------------------------------------------------
    if args.model == 'classification2' and args.predicted_attribute == 'age':
        # Create yound and old classes
        Ytrain = Ytrain.to_frame()
        Ytest = Ytest.to_frame()
        # Yvalidate = Yvalidate.to_frame()
        conditions_test = [Ytest <=30]
        conditions_train = [Ytrain <=30]
        conditions_validate = [Yvalidate<=30]
        # Create two classes (young and old)
        # x < 30 is considered young: Class = 0
        # x > 60 is considered old: Class = 1
        choices = [0]
        Ytest['class'] = np.select(conditions_test, choices, default=1)
        Ytrain['class'] = np.select(conditions_train, choices, default=1)
        Yvalidate['class'] = np.select(conditions_validate, choices, default=1)


    # Transform pandas into numpy arrays (no nneed to do it if you are scaling
    # the results)
    if args.model == 'regression' or args.predicted_attribute != 'age':
        Ytrain = Ytrain.values
        Ytest = Ytest.values
        # Yvalidate = Yvalidate.values
    elif args.model == 'classification':
        pass
    else:
        Ytrain = Ytrain['class'].values
        Ytest = Ytest['class'].values
        Yvalidate = Yvalidate['class'].values




    print(' ')
    print('-----------------------------------------------------------------')
    print('Fit TPOT:')
    print('-----------------------------------------------------------------')
    print('Running %s  analyis with TPOT' %args.model)
    if args.model == 'classification' or args.model == 'classification2':
        tpot = TPOTClassifier(generations=args.generations,
                             population_size=args.population_size,
                             offspring_size=args.offspring_size,
                             mutation_rate=args.mutation_rate,
                             crossover_rate=args.crossover_rate,
                             n_jobs=args.njobs,
                             cv=args.cv,
                             verbosity=3,
                             random_state=args.random_seed,
                             config_dict=tpot_config,
                             scoring='accuracy',
                             periodic_checkpoint_folder=best_pipe_paths,
                             use_dask=False,
                            )
    elif args.model == 'regression':
        tpot = ExtendedTPOTRegressor(generations=args.generations,
                             population_size=args.population_size,
                             offspring_size=args.offspring_size,
                             mutation_rate=args.mutation_rate,
                             crossover_rate=args.crossover_rate,
                             n_jobs=args.njobs,
                             cv=args.cv,
                             verbosity=3,
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
    t_training = time.process_time()
    if args.model == 'regression':
        tpot.fit(Xtrain_scaled, Ytrain, Xtest_scaled, Ytest)
    else:
        tpot.fit(Xtrain_scaled, Ytrain)
    print('TPOT trainig - Elapsed time in seconds:')
    elapsed_time = time.process_time() - t_training
    print('%.03f' %elapsed_time)
    print('')

    t_test = time.process_time()
    tpot_score_test = tpot.score(Xtest_scaled, Ytest)
    print('TPOT test - Elapsed time in seconds:')
    elapsed_time = time.process_time() - t_test
    print('%.03f' %elapsed_time)
    print('')

    print('Test score using optimal model: %.3f ' % tpot_score_test)
    tpot.export(os.path.join(output_path, 'tpot_brain_age_pipeline.py'))
    print('Done TPOT analysis!')


    print('-----------------------------------------------------------------')
    print('Results Analysis:')
    print('-----------------------------------------------------------------')

    # Find the class for the predicted subjects (on the test dataset)
    tpot_predictions = tpot.predict(Xtest_scaled)

    if args.model == 'regression':
        print('Number of models analysed: %d' % len(tpot.predictions))
        repeated_idx = np.argwhere(
              [np.array_equal(np.repeat(tpot.predictions[i][0], len(tpot.predictions[i])),
                              tpot.predictions[i]) for i in range(len(tpot.predictions))])
        print('Index of the models with the same prediction for all subjects: ' + str(np.squeeze(repeated_idx)))
        # tpot_predictions = np.delete(np.array(tpot.predictions), np.squeeze(repeated_idx), axis=0)

        # Dump tpot.pipelines and evaluated objects
        print('Dump predictions, evaluated pipelines and sklearn objects')
        tpot_save = {}
        tpot_pipelines = {}
        # List of predictions using the best model
        tpot_save['predictions'] = tpot.predictions
        # List of evaluated individuals
        tpot_save['evaluated_individuals_'] = tpot.evaluated_individuals_
        # Best pipeline at the end of the genetic algorithm
        tpot_save['fitted_pipeline'] = tpot.fitted_pipeline_
        # List of evaluated invidivuals per generation
        tpot_save['evaluated_individuals'] = tpot.evaluated_individuals
        # Dictionary containing all pipelines in the TPOT Pareto Front
        tpot_save['pareto_pipelines'] = tpot.pareto_front_fitted_pipelines_
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

    if args.model == 'classification' or args.model == 'classification2':
        # Create confusion matrix on test data
        if args.predicted_attribute == 'Prospective_memory':
            class_name = np.array(['Not_remembered', 'Second_Attempt',
                                    'First_Attempt'], dtype='U30')
        elif args.predicted_attribute == 'gender':
            class_name = np.array(['male', 'female'], dtype='U10')
        else:
            class_name = np.array(['young', 'old', 'adult'], dtype='U10')
        ax, cm_test = plot_confusion_matrix(Ytest, tpot_predictions, classes=class_name,
                              title='Confusion Matrix', normalize=True)
        plt.savefig(os.path.join(output_path, 'confusion_matrix_tpot_test.eps'))

        # Predict age for the validation dataset
        tpot_score_validation = tpot.score(Xvalidate_scaled, Yvalidate)
        print('Validation score using optimal model: %.3f' %tpot_score_validation)
        tpot_predictions_val = tpot.predict(Xvalidate_scaled)
        ax, cm_validate = plot_confusion_matrix(Yvalidate, tpot_predictions_val,
                                                classes=class_name,
                                                 normalize=True)
        plt.savefig(os.path.join(output_path, 'confusion_matrix_tpot_val.eps'))

        tpot_save = {}
        # List of variables to save
        tpot_save['confusion_matrix_test'] = cm_test
        tpot_save['confusion_matrix_validatate'] = cm_validate
        tpot_save['evaluated_individuals_'] = tpot.evaluated_individuals_
        tpot_save['fitted_model'] = tpot.fitted_pipeline_ # best pipeline
        tpot_save['score_test'] = tpot_score_test
        tpot_save['score_validation'] = tpot_score_validation
        # Best pipeline at the end of the genetic algorithm

        # Dump results
        joblib.dump(tpot_save, os.path.join(output_path, 'tpot_%s_%s_%03dgen.dump')
                                     %(args.dataset, args.config_dict,
                                       args.generations))
    if args.gui:
        plt.show()

