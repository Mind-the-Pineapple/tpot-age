# Custom defined list of Gaussian Process regression models to be used by TPOT
import numpy as np
import pdb
from itertools import product

from skrvm import RVR
from xgboost import XGBRegressor

# Define list of Kernels
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)
# The hyperparameters for the GPR, will be optimised during fitting
kernels = [RBF(), RationalQuadratic(), ExpSineSquared(), Matern()]

tpot_config_gpr = {
    'sklearn.gaussian_process.GaussianProcessRegressor': {
        'kernel': kernels,
        'random_state': [42],
        'alpha': np.arange(1e-2, 10, 30)
    },
    'skrvm.RVR': {
           'kernel': kernels,
           'alpha': [1e-10, 1e-06, 1e-02, 1],
           'beta': [1e-10, 1e-06, 1e-02, 1],

                 },
    'sklearn.svm.LinearSVR': {
           'loss': ["epsilon_insensitive",
                    "squared_epsilon_insensitive"],
           'dual': [True, False],
           'random_state': [42],
           'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
           'C': [2 ** -6, 2 ** -5, 2 ** -4, 2 ** -3, 2 ** -2, 2 ** -1, 2 ** 0,
                 2 ** 1.],
           'epsilon': [1e-4, 1e-3, 1e-2, 1e-1, 1.]
    },
    'sklearn.ensemble.RandomForestRegressor': {
           'n_estimators': [100],
           'max_features': np.arange(0.05, 1.01, 0.05),
           'random_state': [42],
           'min_samples_split': range(2, 21),
           'min_samples_leaf':  range(1, 21),
           'bootstrap': [True, False]
    },
    'sklearn.neighbors.KNeighborsRegressor': {
         'n_neighbors': range(1,101),
          'weights': ["uniform", "distance"],
          'p': [1, 2]
    },
    'sklearn.linear_model.LinearRegression':{
        },
    'sklearn.linear_model.Ridge': {
        'alpha': [1.0, 10.0, 100.0],
        'random_state': [42],
        },
    # Additional models
    'sklearn.linear_model.ElasticNetCV': {
        'l1_ratio': np.arange(0.0, 1.01, 0.05),
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'random_state': [42],
    },
    'sklearn.ensemble.ExtraTreesRegressor': {
        'n_estimators': [100],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'random_state': [42],
        'bootstrap': [True, False]
    },
    'sklearn.linear_model.LassoLarsCV': {
           'normalize': [True, False]
    },
        'sklearn.tree.DecisionTreeRegressor': {
           'max_depth': range(1, 11),
           'min_samples_split': range(2, 21),
           'random_state': [42],
           'min_samples_leaf': range(1, 21)
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
                    'sklearn.feature_selection.f_regression': None}
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
