from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Set up Dask
# Note: After instantiating the Client you can open
# http://localhost:8787/status to see the dashboard of workers
# To see the dashboard bokeh needs to be installed on your enviroment
from dask.distributed import Client
client = Client(threads_per_worker=1)
client

# Create Data
digits = load_digits()

# To ensure the example runs quickly, we'll make the training dataset relatively
# small.
X_train, X_test, y_train, y_test = train_test_split(
                                                        digits.data,
                                                        digits.target,
                                                        train_size=0.05,
                                                        test_size=0.95,
                                                   )
# Using Dask
# scale up: Increase the TPOT parameters like population_size, generations
tp = TPOTClassifier(
                        generations=2,
                        population_size=10,
                        cv=2,
                        n_jobs=-1,
                        random_state=0,
                        verbosity=2,
                        use_dask=True
                   )
tp.fit(X_train, y_train)
