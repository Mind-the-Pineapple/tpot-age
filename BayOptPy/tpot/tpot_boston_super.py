from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from tempfile import mkdtemp
from tpot.base import TPOTBase
from stopit import threading_timeoutable, TimeoutException
from tpot.operator_utils import set_sample_weight
from sklearn.utils import indexable
from sklearn.base import clone, is_classifier
from sklearn.model_selection._split import check_cv
from sklearn.metrics.scorer import check_scoring
import warnings
from sklearn.model_selection._validation import _fit_and_score

from functools import partial
from multiprocessing import cpu_count
from sklearn.externals.joblib import Parallel, delayed

from tpot.config.regressor import regressor_config_dict


class ExtendedTPOTBase(TPOTBase):

    def _fit_init(self):
        super()._fit_init()
        # Initialise list to save the predictions and pipelines analysed by TPOT
        self.predictions = []
        self.pipelines = []


    def fit(self, features, target, features_test):
        # Pass the features of the test set so that they can be used for the predictions
        self.features_test = features_test
        self = super().fit(features, target)


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

                    parallel = Parallel(n_jobs=self._n_jobs, verbose=0, pre_dispatch='2*n_jobs')
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


class ExtendedTPOTRegressor(ExtendedTPOTBase):
    """TPOT estimator for regression problems."""

    scoring_function = 'neg_mean_squared_error'  # Regression scoring
    default_config_dict = regressor_config_dict  # Regression dictionary
    classification = False
    regression = True

housing = load_boston()
X_train, X_test, y_train, y_test = \
train_test_split(housing.data, housing.target, train_size=0.75, test_size=0.25)
# used scoring
scoring = 'neg_mean_absolute_error'
tpot_config = {
    'sklearn.linear_model.ElasticNetCV': {
    'l1_ratio': np.arange(0.0, 1.01),
    'tol': [1e-5]
    },
    'sklearn.neighbors.KNeighborsRegressor': {
    'n_neighbors': range(1,2),
    'weights': ["uniform", "distance"],
    'p': [1, 2]
    },
    # preprocessing
    'sklearn.decomposition.PCA': {
    'svd_solver': ['randomized'],
    'iterated_power': range(1,2)
    }
}
# create a directory where to cache the results
cachedir = mkdtemp()
tpot = ExtendedTPOTRegressor(generations=5,
                     population_size=50,
                     verbosity=2,
                     random_state=42,
                     config_dict='TPOT light',
                     scoring=scoring
                     )
tpot.fit(X_train, y_train, X_test)
print('Test score using optimal model: %f ' %tpot.score(X_test, y_test))
tpot.export('tpot_boston_pipeline.py')

# Cross correlate the predictions
corr_matrix = np.corrcoef(tpot.predictions)
import seaborn as sns
from matplotlib.pylab import plt
sns.heatmap(corr_matrix)
plt.show()


