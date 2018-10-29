from tpot import TPOTRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from deap import creator
from sklearn.model_selection import cross_val_score
import numpy as np

housing = load_boston()
X_train, X_test, y_train, y_test = \
train_test_split(housing.data, housing.target, train_size=0.75, test_size=0.25)
# used scoring
scoring = 'neg_mean_absolute_error'

tpot = TPOTRegressor(generations=5,
                     population_size=50,
                     verbosity=3,
                     random_state=42,
                     config_dict='TPOT light',
                     scoring=scoring
                     )
tpot.fit(X_train, y_train)
print('Test score using optimal model: %f ' %tpot.score(X_test, y_test))
tpot.export('tpot_boston_pipeline.py')

# Evaluate analysed pipelines
# print the top two values of the pipeline dictionary
print(dict(list(tpot.evaluated_individuals_.items())[0:2]))
# print a pipeline and its values
pipeline_str = list(tpot.evaluated_individuals_.keys())[0]
print(pipeline_str)
print(tpot._evaluated_individuals[pipeline_str])
# convert pipeline string to scikit-learn pipeline object
optimized_pipeline = creator.Individual.from_string(pipeline_str, tpot._pset) # deap object
fitted_pipeline = tpot._toolbox.compile(expr=optimized_pipeline ) # scikit-learn pipeline object
# print scikit-learn pipeline object
print(fitted_pipeline)
# Fix random state when the operator allows  (optional) just for get consistent CV score
tpot._set_param_recursive(fitted_pipeline.steps, 'random_state', 42)
scores = cross_val_score(fitted_pipeline, X_train, y_train, cv=5, scoring=scoring, verbose=0)
print(np.mean(scores))
print(tpot._evaluated_individuals[pipeline_str][1])

print('Done')