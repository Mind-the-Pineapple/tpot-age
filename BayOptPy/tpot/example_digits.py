from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-njobs',
                   dest='njobs',
                   type=int,
                   required=True)
args = parser.parse_args()

digits = load_digits()

random_seed = 0

X_train, X_test, y_train, y_test = train_test_split(digits.data,
                                                    digits.target,
                                                    train_size=0.75,
                                                    test_size=0.25,
                                                    random_state=random_seed)

tpot = TPOTClassifier(generations=5,
                      population_size=40,
                      cv=5,
                      n_jobs=args.njobs,
                      random_state=random_seed,
                      verbosity=2,
                      use_dask=False)

tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
