from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()

X_train, X_test, y_train, y_test = train_test_split(digits.data,
                                                    digits.target,
                                                    train_size=0.75,
                                                    test_size=0.25)

tpot = TPOTClassifier(generations=5,
                      population_size=40,
                      cv=5,
                      n_jobs=1,
                      random_state=0,
                      verbosity=2,
                      use_dask=False)

tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
