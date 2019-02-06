from tpot import TPOTRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from deap import creator
from sklearn.model_selection import cross_val_score, cross_val_predict
import numpy as np
from tempfile import mkdtemp
from shutil import rmtree

random_seed = 42
housing = load_boston()
X_train, X_test, y_train, y_test = \
train_test_split(housing.data, housing.target, train_size=0.75, test_size=0.25, random_state=random_seed)
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
tpot = TPOTRegressor(generations=5,
                     population_size=50,
                     verbosity=2,
                     random_state=random_seed,
                     config_dict='TPOT light',
                     scoring=scoring
                     )
tpot.fit(X_train, y_train)
print('Test score using optimal model: %f ' %tpot.score(X_test, y_test))
tpot.export('BayOptPy/tpot/debug/tpot_boston_pipeline.py')
# get the list of models analysed
analysed_models = list(tpot.evaluated_individuals_.items())
predicted_age = {}

for model in analysed_models:
    model_name = model[0]
    model_info = model[1]

    # fit the data
    optimised_pipeline = creator.Individual.from_string(model_name, tpot._pset)
    # Transform the tree expression int a callable function
    fitted_pipeline = tpot._toolbox.compile(expr=optimised_pipeline)
    tpot._set_param_recursive(fitted_pipeline.steps, 'random_state', random_seed)
    predicted_age[model_name] = {}
    predicted_age[model_name]['age'] = cross_val_predict(fitted_pipeline, X_test, y_test, cv=5)
# remove the cached directory
rmtree(cachedir)
print('Done')

# Evaluate analysed pipelines
# print the top two values of the pipeline dictionary
print(dict(list(tpot.evaluated_individuals_.items())[0:2]))
# print a pipeline and its values
pipeline_str = list(tpot.evaluated_individuals_.keys())[0]
print(pipeline_str)
print(tpot.evaluated_individuals_[pipeline_str])
# convert pipeline string to scikit-learn pipeline object
optimized_pipeline = creator.Individual.from_string(pipeline_str, tpot._pset) # deap object
fitted_pipeline = tpot._toolbox.compile(expr=optimized_pipeline) # scikit-learn pipeline object
# print scikit-learn pipeline object
print(fitted_pipeline)
# Fix random state when the operator allows  (optional) just for get consistent CV score
tpot._set_param_recursive(fitted_pipeline.steps, 'random_state', random_seed)
scores = cross_val_score(fitted_pipeline, X_train, y_train, cv=5, scoring=scoring, verbose=0)
print(np.mean(scores))
print(tpot.evaluated_individuals[pipeline_str][1])

