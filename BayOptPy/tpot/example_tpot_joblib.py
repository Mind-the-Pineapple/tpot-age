import multiprocessing

# Set up multiprocessing
if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')
    from tpot import TPOTClassifier
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    # Create Data
    digits = load_digits()
    random_seed = 0

    # To ensure the example runs quickly, we'll make the training dataset relatively
    # small.
    X_train, X_test, y_train, y_test = train_test_split(digits.data,
                                                        digits.target,
                                                        train_size=0.75,
                                                        test_size=0.25,
                                                        random_state=random_seed)
    # Using Dask
    # scale up: Increase the TPOT parameters like population_size, generations
    tp = TPOTClassifier(generations=5,
                        population_size=40,
                        cv=5,
                        n_jobs=4,
                        random_state=random_seed,
                        verbosity=2,
                        use_dask=False)
    tp.fit(X_train, y_train)
    print(tp.score(X_test, y_test))
