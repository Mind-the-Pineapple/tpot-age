# Custom defined list of Gaussian Process regression models to be used by TPOT
import numpy as np

from skrvm import RVC

tpot_config_gpr = {
                  # Classifiers
                  'skrvm.RVC': {
                                'alpha': [1e-10, 1e-06, 1e-02, 1],
                                'kernel': ['rbf', 'linear'],
                               },
                  'sklearn.svm.SVC':{
                               'C': [2 ** -6, 2 ** -5, 2 ** -4,
                                     2 ** -3, 2 ** -2, 2 ** -1,
                                     2 ** 0, 2 ** 1.],
                                'kernel': ['rbf', 'linear'],
                                'random_state': 42
                                    },
                  'sklearn.ensemble.RandomForestClassifier': {
                                'n_estimators': [100],
                                'max_features': np.arange(0.05,1.01, 0.05),
                                'random_state': [42],
                                'min_samples_split': range(2, 21),
                                'min_samples_leaf': range(1,21),
                                'bootstrap': [True,False]
                                    },
                  'sklearn.naive_bayes.GaussianNB': { },

                  'sklearn.naive_bayes.BernoulliNB': {
                                'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
                                'fit_prior': [True, False]
                                     },
                  'sklearn.naive_bayes.MultinomialNB': {
                                 'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
                                 'fit_prior':
                                 [True, False]
                                 },
                   'sklearn.ensemble.AdaBoostClassifier': {
                                'n_estimators' : range(10, 100, 10),
                                'learning_rate': np.arange(.1, 1.01, .1),
                                'random_state': [42]
                                  },
                  'sklearn.tree.DecisionTreeClassifier': {
                                 'criterion': ["gini", "entropy"],
                                 'max_depth': range(1, 11),
                                 'min_samples_split':range(2, 21),
                                 'min_samples_leaf':range(1, 21)
                                     },
                  'sklearn.neighbors.KNeighborsClassifier': {
                                  'n_neighbors': range(1, 101),
                                  'weights': ["uniform", "distance"],
                                  'p': [1, 2]
                                   },
                  'sklearn.linear_model.LogisticRegression': {
                                   'penalty':  ["l1", "l2"],
                                   'C': [1e-4, 1e-3, 1e-2,
                                         1e-1, 0.5, 1.,
                                         5., 10., 15.,
                                         20., 25.],
                                    'dual': [True, False]
                                       },

###############################################################################
## Preprocessors
################################################################################
    'sklearn.kernel_approximation.Nystroem': {
            'kernel': ['rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 'poly',
                       'linear', 'additive_chi2', 'sigmoid'],
            'gamma': np.arange(0.0, 1.01, 0.05),
            'random_state': [42],
            'n_components': range(1, 11)
    },
        'sklearn.kernel_approximation.RBFSampler': {
            'gamma': np.arange(0.0, 1.01, 0.05),
           'random_state': [42],
    },
        'sklearn.preprocessing.Binarizer': {
           'threshold': np.arange(19.0, 89.0, 1)
                                                },
#################################################################################
### Feature Selection
#################################################################################

        'sklearn.decomposition.PCA':{
            'svd_solver': ['randomized'],
            'iterated_power':range(1,11)
    },
        'sklearn.decomposition.FastICA':{
                'tol': np.arange(0.0, 2.02, 0.05)
    },
        'sklearn.feature_selection.SelectFwe':{
                'alpha': np.arange(0, 0.05, 0.001),
                'score_func': {
                    'sklearn.feature_selection.f_regression': None}
    },
        'sklearn.feature_selection.SelectPercentile': {
                'percentile': range(1, 100),
                'score_func': {
                    'sklearn.feature_selection.f_classif': None}
    },
        'sklearn.feature_selection.VarianceThreshold':{
             'threshold': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1,
                            0.2] },
###############################################################################
## Feature Combination
################################################################################
            'sklearn.cluster.FeatureAgglomeration': {
                 'linkage': ['ward', 'complete', 'average'],
                 'affinity': ['euclidean', 'l1', 'l2',
                              'manhattan', 'cosine'] },
}
