from tpot import TPOTRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

random_seed = 42
housing = load_boston()
X_train, X_test, y_train, y_test = \
train_test_split(housing.data, housing.target, train_size=0.75, test_size=0.25,
                random_state=random_seed)
scoring = 'neg_mean_absolute_error'

tpot = TPOTRegressor(generations=100,
                     population_size=50,
                     verbosity=2,
                     random_state=random_seed,
                     config_dict='TPOT light',
                     scoring=scoring
                     )
tpot.fit(X_train, y_train)
print('Test score using optimal model: %f ' %tpot.score(X_test, y_test))
tpot.export('tpot_boston_pipeline.py')
