import numpy as np
from tpot.base import TPOTBase
from stopit import threading_timeoutable, TimeoutException
from tpot.operator_utils import set_sample_weight
from sklearn.utils import indexable
from sklearn.base import clone, is_classifier
from sklearn.model_selection._split import check_cv
from sklearn.metrics.scorer import check_scoring
import warnings
from sklearn.model_selection._validation import _fit_and_score
from sklearn.gaussian_process.kernels import (RBF, Matern,
                                             RationalQuadratic,
                                             ExpSineSquared, DotProduct,
                                             ConstantKernel)

from functools import partial
from multiprocessing import cpu_count
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from datetime import datetime
import random
from tqdm import tqdm
from deap import tools
from tpot.gp_deap import initialize_stats_dict, varOr
import os
import errno
import pickle


from BayOptPy.helperfunctions import get_all_random_seed_paths
from tpot.config.regressor import regressor_config_dict
from tpot.export_utils import generate_pipeline_code, export_pipeline, expr_to_tree



class ExtendedTPOTBase(TPOTBase):
    def __init__(self, generations=100, population_size=100,
                 offspring_size=None, mutation_rate=0.9, crossover_rate=0.1,
                 scoring=None, cv=5, subsample=1.0, n_jobs=1,
                 max_time_mins=None, max_eval_time_mins=5, random_state=None,
                 config_dict=None, warm_start=False, memory=None,
                 use_dask=False, periodic_checkpoint_folder=None,
                 early_stop=None, verbosity=0, disable_update_check=False,
                 debug=False, analysis=None):
        super().__init__(generations, population_size,
                         offspring_size, mutation_rate,
                         crossover_rate, scoring, cv, subsample,
                         n_jobs, max_time_mins, max_eval_time_mins,
                         random_state, config_dict, warm_start,
                         memory, use_dask,
                         periodic_checkpoint_folder, early_stop,
                         verbosity, disable_update_check)
        self.debug = debug
        # specify type of analysis being used
        self.analysis = analysis
        # Set random seed
        if random_state is not None:
            print('Setting random seed')
            np.random.seed(random_state)

    def _fit_init(self):
        super()._fit_init()
        # Initialise list to save the predictions and pipelines analysed by TPOT
        self.predictions = []
        self.pipelines = []
        self._exported_pipeline_text = []
        # Save training sample on the TPOT Object
        self.features = None
        self.target = None

        # Add the Gaussian kernels so that they can be used by TPOT
        self.operators_context['RBF'] = eval('RBF')
        self.operators_context['Matern'] = eval('Matern')
        self.operators_context['RationalQuadratic'] = eval('RationalQuadratic')
        self.operators_context['ExpSineSquared'] = eval('ExpSineSquared')
        self.operators_context['DotProduct'] = eval('DotProduct')
        self.operators_context['ConstantKernel'] = eval('ConstantKernel')


    def fit(self, features, target, features_test, target_test, sample_weight=None, groups=None):
        # Pass the features of the test set so that they can be used for the predictions
        self.features_test = features_test
        """Fit an optimized machine learning pipeline.

        Uses genetic programming to optimize a machine learning pipeline that
        maximizes score on the provided features and target. Performs internal
        k-fold cross-validaton to avoid overfitting on the provided data. The
        best pipeline is then trained on the entire set of provided samples.

        Parameters
        ----------
        features: array-like {n_samples, n_features}
            Feature matrix

            TPOT and all scikit-learn algorithms assume that the features will be numerical
            and there will be no missing values. As such, when a feature matrix is provided
            to TPOT, all missing values will automatically be replaced (i.e., imputed) using
            median value imputation.

            If you wish to use a different imputation strategy than median imputation, please
            make sure to apply imputation to your feature set prior to passing it to TPOT.
        target: array-like {n_samples}
            List of class labels for prediction
        sample_weight: array-like {n_samples}, optional
            Per-sample weights. Higher weights indicate more importance. If specified,
            sample_weight will be passed to any pipeline element whose fit() function accepts
            a sample_weight argument. By default, using sample_weight does not affect tpot's
            scoring functions, which determine preferences between pipelines.
        groups: array-like, with shape {n_samples, }, optional
            Group labels for the samples used when performing cross-validation.
            This parameter should only be used in conjunction with sklearn's Group cross-validation
            functions, such as sklearn.model_selection.GroupKFold

        Returns
        -------
        self: object
            Returns a copy of the fitted TPOT object

        """
        self._fit_init()

        self.features, self.target = self._check_dataset(features, target, sample_weight)
        # save true prections from the target test
        self.target_test = target_test

        # Randomly collect a subsample of training samples for pipeline optimization process.
        if self.subsample < 1.0:
            features, _, target, _ = train_test_split(features, target, train_size=self.subsample, random_state=self.random_state)
            # Raise a warning message if the training size is less than 1500 when subsample is not default value
            if features.shape[0] < 1500:
                print(
                    'Warning: Although subsample can accelerate pipeline optimization process, '
                    'too small training sample size may cause unpredictable effect on maximizing '
                    'score in pipeline optimization process. Increasing subsample ratio may get '
                    'a more reasonable outcome from optimization process in TPOT.'
                )

        # Set the seed for the GP run
        if self.random_state is not None:
            random.seed(self.random_state)  # deap uses random
            np.random.seed(self.random_state)

        self._start_datetime = datetime.now()
        self._last_pipeline_write = self._start_datetime
        self._toolbox.register('evaluate', self._evaluate_individuals, features=features, target=target, sample_weight=sample_weight, groups=groups)

        # assign population, self._pop can only be not None if warm_start is enabled
        if self._pop:
            pop = self._pop
        else:
            pop = self._toolbox.population(n=self.population_size)

        def pareto_eq(ind1, ind2):
            """Determine whether two individuals are equal on the Pareto front.

            Parameters
            ----------
            ind1: DEAP individual from the GP population
                First individual to compare
            ind2: DEAP individual from the GP population
                Second individual to compare

            Returns
            ----------
            individuals_equal: bool
                Boolean indicating whether the two individuals are equal on
                the Pareto front

            """
            return np.allclose(ind1.fitness.values, ind2.fitness.values)

        # Generate new pareto front if it doesn't already exist for warm start
        if not self.warm_start or not self._pareto_front:
            self._pareto_front = tools.ParetoFront(similar=pareto_eq)

        # Set lambda_ (offspring size in GP) equal to population_size by default
        if not self.offspring_size:
            self._lambda = self.population_size
        else:
            self._lambda = self.offspring_size

        # Start the progress bar
        if self.max_time_mins:
            total_evals = self.population_size
        else:
            total_evals = self._lambda * self.generations + self.population_size

        self._pbar = tqdm(total=total_evals, unit='pipeline', leave=False,
                          disable=not (self.verbosity >= 2), desc='Optimization Progress')

        try:
            with warnings.catch_warnings():
                self._setup_memory()
                warnings.simplefilter('ignore')

                pop, _ = extendedeaMuPlusLambda(
                    population=pop,
                    toolbox=self._toolbox,
                    mu=self.population_size,
                    lambda_=self._lambda,
                    cxpb=self.crossover_rate,
                    mutpb=self.mutation_rate,
                    ngen=self.generations,
                    pbar=self._pbar,
                    halloffame=self._pareto_front,
                    verbose=self.verbosity,
                    per_generation_function=self._check_periodic_pipeline,
                    debug=self.debug,
                    random_seed=self.random_state,
                    analysis=self.analysis,
                )

            # store population for the next call
            if self.warm_start:
                self._pop = pop

        # Allow for certain exceptions to signal a premature fit() cancellation
        except (KeyboardInterrupt, SystemExit, StopIteration) as e:
            if self.verbosity > 0:
                self._pbar.write('', file=self._file)
                self._pbar.write('{}\nTPOT closed prematurely. Will use the current best pipeline.'.format(e),
                                 file=self._file)
        finally:
            # keep trying 10 times in case weird things happened like multiple CTRL+C or exceptions
            attempts = 10
            for attempt in range(attempts):
                try:
                    # Close the progress bar
                    # Standard truthiness checks won't work for tqdm
                    if not isinstance(self._pbar, type(None)):
                        self._pbar.close()

                    self._update_top_pipeline()
                    self._summary_of_best_pipeline(features, target)
                    # Delete the temporary cache before exiting
                    self._cleanup_memory()
                    break

                except (KeyboardInterrupt, SystemExit, Exception) as e:
                    # raise the exception if it's our last attempt
                    if attempt == (attempts - 1):
                        raise e
            return self


    def _evaluate_individuals(self, individuals, features, target, sample_weight=None, groups=None):
        operator_counts, eval_individuals_str, sklearn_pipeline_list, stats_dicts = self._preprocess_individuals(individuals)

        # Make the partial function that will be called below
        partial_wrapped_cross_val_score = partial(
            _wrapped_cross_val_score,
            features=features,
            target=target,
            cv=self.cv,
            scoring_function=self.scoring_function,
            sample_weight=sample_weight,
            groups=groups,
            timeout=max(int(self.max_eval_time_mins * 60), 1),
            use_dask=self.use_dask,
            predictions=self.predictions,
            pipelines=self.pipelines,
            features_test=self.features_test
        )

        result_score_list = []
        # Don't use parallelization if n_jobs==1
        if self._n_jobs == 1 and not self.use_dask:
            for sklearn_pipeline in sklearn_pipeline_list:
                self._stop_by_max_time_mins()
                val = partial_wrapped_cross_val_score(sklearn_pipeline=sklearn_pipeline)
                result_score_list = self._update_val(val, result_score_list)
        else:
            if self.use_dask:
                import dask

                result_score_list = [
                    partial_wrapped_cross_val_score(sklearn_pipeline=sklearn_pipeline)
                    for sklearn_pipeline in sklearn_pipeline_list
                ]

                self.dask_graphs_ = result_score_list
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    result_score_list = list(dask.compute(*result_score_list))

                self._update_pbar(len(result_score_list))

            else:
                # chunk size for pbar update
                # chunk size is min of cpu_count * 2 and n_jobs * 4
                chunk_size = min(cpu_count()*2, self._n_jobs*4)

                for chunk_idx in range(0, len(sklearn_pipeline_list), chunk_size):
                    self._stop_by_max_time_mins()

                    parallel = Parallel(n_jobs=self._n_jobs,
                                        backend='loky',
                                        prefer='processes',
                                        verbose=0, pre_dispatch='2*n_jobs')
                    tmp_result_scores = parallel(
                        delayed(partial_wrapped_cross_val_score)(sklearn_pipeline=sklearn_pipeline)
                        for sklearn_pipeline in sklearn_pipeline_list[chunk_idx:chunk_idx + chunk_size])
                    # update pbar
                    for val in tmp_result_scores:
                        result_score_list = self._update_val(val, result_score_list)

        self._update_evaluated_individuals_(result_score_list, eval_individuals_str, operator_counts, stats_dicts)

        """Look up the operator count and cross validation score to use in the optimization"""
        return [(self.evaluated_individuals_[str(individual)]['operator_count'],
                 self.evaluated_individuals_[str(individual)]['internal_cv_score'])
                for individual in individuals]

    def _check_periodic_pipeline(self, gen):
        """If enough time has passed, save a new optimized pipeline.
        Currently used in the per generation hook in the optimization
        loop.
        Parameters
        ----------
        gen: int Generation number

        Returns
        -------
        None
        """
        self._update_top_pipeline()
        log = {}
        if self.periodic_checkpoint_folder is not None:
            total_since_last_pipeline_save =(datetime.now()  - self._last_pipeline_write).total_seconds()
            filename = None
            if total_since_last_pipeline_save > self._output_best_pipeline_period_seconds:
                self._last_pipeline_write = datetime.now()
                log, filename = self._save_periodic_pipeline(gen, log)

        if self.early_stop is not None:
            if self._last_optimized_pareto_front_n_gens >= self.early_stop:
                raise StopIteration("The optimized pipeline was not improved after evaluating {} more generations.  "
                                    "Will end the optimization process.\n".format(self.early_stop))
        # save log only if it has not been saved previously. Use the previusly saved python script to check wheater to
        # save the log
        if filename is not None:
            log_path = os.path.join(self.periodic_checkpoint_folder, 'pipeline_log_gen_%03d.pckl' %gen)
            with open(log_path, 'wb') as handle:
                pickle.dump(log, handle)

    def _save_periodic_pipeline(self, gen, log):
        try:
            #self._create_periodic_checkpoint_folder()
            for pipeline, pipeline_scores in zip(self._pareto_front.items,
                                                 reversed(self._pareto_front.keys)):
                idx = self._pareto_front.items.index(pipeline)
                pareto_front_pipeline_score = pipeline_scores.wvalues[1]
                sklearn_pipeline_str = generate_pipeline_code(expr_to_tree(pipeline, self._pset), self.operators)
                to_write = export_pipeline(pipeline, self.operators, self._pset,
                                           self._imputed,
                                           pareto_front_pipeline_score,
                                           self.random_state)

                # fit the pipeline again and get the test score
                sklearn_pipeline = self._toolbox.compile(expr=pipeline)
                sklearn_pipeline.fit(self.features, self.target)
                ypredict = sklearn_pipeline.predict(self.features_test)
                from sklearn.metrics import mean_absolute_error
                mae = mean_absolute_error(self.target_test, ypredict)

                # dump a pickle with current pareto value and the pipeline name
                log['pipeline_name'] = sklearn_pipeline_str
                log['pipeline_score'] = pipeline_scores.wvalues[1]
                log['pipeline_test_mae'] = mae
                # dont export a pipeline you had
                if self._exported_pipeline_text.count(sklearn_pipeline_str):
                    self._update_pbar(pbar_num=0, pbar_msg='Periodic pipeline was not saved, probably saved before...')
                    filename = None
                else:
                    filename = os.path.join(self.periodic_checkpoint_folder,
                                            'pipeline_gen_{}_idx_{}_{}.py'.format(gen, idx, datetime.now().strftime('%Y.%m.%d_%H-%M-%S')))
                    self._update_pbar(pbar_num=0, pbar_msg='Saving periodic pipeline from pareto front to {}'.format(filename))
                    with open(filename, 'w') as output_file:
                        output_file.write(to_write)
                    self._exported_pipeline_text.append(sklearn_pipeline_str)

        except Exception as e:
            self._update_pbar(pbar_num=0, pbar_msg='Failed saving periodic pipeline,   exception:\n{}'.format(str(e)[:250]))

        return log, filename

    def _create_periodic_checkpoint_folder(self):
        try:
            os.makedirs(self.periodic_checkpoint_folder)
            self._update_pbar(pbar_msg='Created new folder to save periodic pipeline: {}'.format(self.periodic_checkpoint_folder))
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(self.speriodic_checkpoint_folder):
                pass # Folder already exists.  User probably created it.
            else:
                raise ValueError('Failed creating the periodic_checkpoint_folder:\n{}'.format(e))


@threading_timeoutable(default="Timeout")
def _wrapped_cross_val_score(sklearn_pipeline, features, target,
                             cv, scoring_function, sample_weight=None,
                             groups=None, use_dask=False, predictions=None, pipelines=None, features_test=None):
    """Fit estimator and compute scores for a given dataset split.

    Parameters
    ----------
    sklearn_pipeline : pipeline object implementing 'fit'
        The object to use to fit the data.
    features : array-like of shape at least 2D
        The data to fit.
    target : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.
    cv: int or cross-validation generator
        If CV is a number, then it is the number of folds to evaluate each
        pipeline over in k-fold cross-validation during the TPOT optimization
         process. If it is an object then it is an object to be used as a
         cross-validation generator.
    scoring_function : callable
        A scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
    sample_weight : array-like, optional
        List of sample weights to balance (or un-balanace) the dataset target as needed
    groups: array-like {n_samples, }, optional
        Group labels for the samples used while splitting the dataset into train/test set
    use_dask : bool, default False
        Whether to use dask
    """
    sample_weight_dict = set_sample_weight(sklearn_pipeline.steps, sample_weight)

    features, target, groups = indexable(features, target, groups)

    cv = check_cv(cv, target, classifier=is_classifier(sklearn_pipeline))
    cv_iter = list(cv.split(features, target, groups))
    scorer = check_scoring(sklearn_pipeline, scoring=scoring_function)

    # save the sklearn predictions. The model is trained with the training set (features) and validated with the test dataset
    # (features_test)
    # Note: because of the way TPOT is built, the fit function is called to see if the model is valid.
    try:
        tmp = sklearn_pipeline.fit(features, target)
        predictions.append(tmp.predict(features_test))
        pipelines.append(sklearn_pipeline)
    except:
        pass

    if use_dask:
        try:
            import dask_ml.model_selection  # noqa
            import dask  # noqa
            from dask.delayed import Delayed
        except ImportError:
            msg = "'use_dask' requires the optional dask and dask-ml depedencies."
            raise ImportError(msg)

        dsk, keys, n_splits = dask_ml.model_selection._search.build_graph(
            estimator=sklearn_pipeline,
            cv=cv,
            scorer=scorer,
            candidate_params=[{}],
            X=features,
            y=target,
            groups=groups,
            fit_params=sample_weight_dict,
            refit=False,
            error_score=float('-inf'),
        )

        cv_results = Delayed(keys[0], dsk)
        scores = [cv_results['split{}_test_score'.format(i)]
                  for i in range(n_splits)]
        CV_score = dask.delayed(np.array)(scores)[:, 0]
        return dask.delayed(np.nanmean)(CV_score)
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                scores = [_fit_and_score(estimator=clone(sklearn_pipeline),
                                         X=features,
                                         y=target,
                                         scorer=scorer,
                                         train=train,
                                         test=test,
                                         verbose=0,
                                         parameters=None,
                                         fit_params=sample_weight_dict)
                          for train, test in cv_iter]
                CV_score = np.array(scores)[:, 0]
                return np.nanmean(CV_score)
        except TimeoutException:
            return "Timeout"
        except Exception as e:
            return -float('inf')


# Modify original eaMuPlusLambda, so that the logbook can be saved. And add the mean and std to the logbook
def extendedeaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen, pbar,
                           stats=None, halloffame=None, verbose=0,
                           per_generation_function=None, debug=False,
                           random_seed=None, analysis=None):
    """This is the :math:`(\mu + \lambda)` evolutionary algorithm.
    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param mu: The number of individuals to select for the next generation.
    :param lambda\_: The number of children to produce at each generation.
    :param cxpb: The probability that an offspring is produced by crossover.
    :param mutpb: The probability that an offspring is produced by mutation.
    :param ngen: The number of generation.
    :param pbar: processing bar
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :param per_generation_function: if supplied, call this function before each generation
                            used by tpot to save best pipeline before each new generation
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution.
    The algorithm takes in a population and evolves it in place using the
    :func:`varOr` function. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evalutions for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varOr` function. The pseudocode goes as follow ::
        evaluate(population)
        for g in range(ngen):
            offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
            evaluate(offspring)
            population = select(population + offspring, mu)
    First, the individuals having an invalid fitness are evaluated. Second,
    the evolutionary loop begins by producing *lambda_* offspring from the
    population, the offspring are generated by the :func:`varOr` function. The
    offspring are then evaluated and the next generation population is
    selected from both the offspring **and** the population. Finally, when
    *ngen* generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.
    This function expects :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox. This algorithm uses the :func:`varOr`
    variation.
    """

    if random_seed == None:
        raise ValueError("No fixed random seed was used!")

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals', 'avg', 'std', 'min', 'max', 'raw'] + (stats.fields if stats else [])

    # Initialize statistics dict for the individuals in the population, to keep track of mutation/crossover operations and predecessor relations
    for ind in population:
        initialize_stats_dict(ind)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]

    fitnesses = toolbox.evaluate(invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    # calculate average fitness for the generation
    # ignore the -inf models
    fitnesses_only = np.array([fitnesses[i][1] for i in range(len(population))])
    n_inf = np.sum(np.isinf(fitnesses_only))
    print('Number of invalid pipelines: %d' %n_inf)
    fitnesses_only = fitnesses_only[~np.isinf(fitnesses_only)]


    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind),
                   avg=np.mean(fitnesses_only), std=np.std(fitnesses_only),
                   min=np.min(fitnesses_only), max=np.max(fitnesses_only),
                   raw=fitnesses_only,
                   **record)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # after each population save a periodic pipeline
        if per_generation_function is not None:
            per_generation_function(gen)

        # Vary the population
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Update generation statistic for all individuals which have invalid 'generation' stats
        # This hold for individuals that have been altered in the varOr function
        for ind in population:
            if ind.statistics['generation'] == 'INVALID':
                ind.statistics['generation'] = gen

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        # update pbar for valid individuals (with fitness values)
        if not pbar.disable:
            pbar.update(len(offspring)-len(invalid_ind))

        fitnesses = toolbox.evaluate(invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        # pbar process
        if not pbar.disable:
            # Print only the best individual fitness
            if verbose == 2:
                high_score = max([halloffame.keys[x].wvalues[1] for x in range(len(halloffame.keys))])
                pbar.write('Generation {0} - Current best internal CV score: {1}'.format(gen, high_score))

            # Print the entire Pareto front
            elif verbose == 3:
                pbar.write('Generation {} - Current Pareto front scores:'.format(gen))
                for pipeline, pipeline_scores in zip(halloffame.items, reversed(halloffame.keys)):
                    pbar.write('{}\t{}\t{}'.format(
                        int(pipeline_scores.wvalues[0]),
                        pipeline_scores.wvalues[1],
                        pipeline
                    )
                    )
                pbar.write('')

        # calculate average fitness for the generation
        # ignore the -inf models
        fitnesses_only = np.array([fitnesses[i][1] for i in range(len(offspring))])
        n_inf = np.sum(np.isinf(fitnesses_only))
        print('Number of invalid pipelines: %d' %n_inf)
        fitnesses_only = fitnesses_only[~np.isinf(fitnesses_only)]

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind),
                       avg=np.mean(fitnesses_only), std=np.std(fitnesses_only),
                       min=np.min(fitnesses_only), max=np.max(fitnesses_only),
                       raw=fitnesses_only,
                       **record)
    # Dump logbook
    import pickle
    import pandas as pd
    deap_df = pd.DataFrame(logbook)
    save_path = get_all_random_seed_paths(analysis, ngen, len(population), debug)
    save_path_df = os.path.join(save_path, 'logbook_rnd_seed%03d.pkl'
                                %random_seed)
    with open(save_path_df, 'wb') as handle:
        pickle.dump(deap_df, handle)
    print('Saved logbook at %s' %save_path_df)

    return population, logbook


class ExtendedTPOTRegressor(ExtendedTPOTBase):
    """TPOT estimator for regression problems."""

    scoring_function = 'neg_mean_squared_error'  # Regression scoring
    default_config_dict = regressor_config_dict  # Regression dictionary
    classification = False
    regression = True
