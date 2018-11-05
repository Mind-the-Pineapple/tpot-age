from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Set up Dask
# Note: After instantiating the Client you can open
# http://localhost:8787/status to see the dashboard of workers
# To see the dashboard bokeh needs to be installed on your enviroment
from sklearn.externals import joblib
import distributed.joblib
from dask.distributed import Client
client = Client(diagnostics_port=8788, processes=False)
client

# Create Data
digits = load_digits()

# To ensure the example runs quickly, we'll make the training dataset relatively
# small.
X_train, X_test, y_train, y_test = train_test_split(digits.data,
                                                    digits.target,
                                                    train_size=0.75,
                                                    test_size=0.25)
# Using Dask
# scale up: Increase the TPOT parameters like population_size, generations.
# Note: When use_dask = True, TPOT will use as manu cores as avaliable on the cluster, regardless of the n_jobs specified
tp = TPOTClassifier(generations=5,
                    population_size=40,
                    cv=5,
                    random_state=0,
                    verbosity=2,
                    use_dask=True)
with joblib.parallel_backend('dask'):
    tp.fit(X_train, y_train)
print(tp.score(X_test, y_test))
