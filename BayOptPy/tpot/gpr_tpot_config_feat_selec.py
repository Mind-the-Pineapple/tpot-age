# Custom defined list of Gaussian Process regression models to be used by TPOT
import numpy as np
import pdb
from itertools import product

from skrvm import RVR

# Define list of Kernels
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)
kernels_rbf = [RBF(length_scale=x) for x in np.arange(0., 1.1, .1)]
kernels_rq = [RationalQuadratic(length_scale=x, alpha=y) for x,y in
              product(np.arange(0., 1.1, .1), np.arange(0.1, 10.1,1))]
kernels_exp = [ExpSineSquared(length_scale=x, periodicity=y) for x,y in
                     product(np.arange(0., 1.1, .1), np.arange(.01, 10.1,1))]
kernels_mat = [Matern(length_scale=x, nu=y) for x,y in
                     product(np.arange(0.1, 1.1,.1), [.5, 1.5, 2.5])]
kernel_dot = [DotProduct(sigma_0=x) for x in np.arange(0., 1, .01)]
kernels = kernels_rbf + kernels_rq + kernels_exp + kernels_mat + kernel_dot

tpot_config_gpr = {
    'sklearn.gaussian_process.GaussianProcessRegressor': {
        'kernel': kernels,
        'random_state': [42],
        'alpha': [1e-100, 1e-20, 1e-12, 1e-10, 1e-5, 1e-3]
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
           'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5.,
                 10., 15., 20., 25.],
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
    'sklearn.neighbors.KNeighborsRegressor': {
         'n_neighbors': range(1,101),
          'weights': ["uniform", "distance"],
          'p': [1, 2]
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
        'sklearn.preprocessing.RobustScaler': {
                                              },
         'sklearn.preprocessing.StandardScaler': {
                                                 },
        'sklearn.kernel_approximation.RBFSampler': {
            'gamma': np.arange(0.0, 1.01, 0.05),
           'random_state': [42],
    },
        'sklearn.preprocessing.MaxAbsScaler': {
                                              },
        'sklearn.preprocessing.MinMaxScaler': {
                                              },
        'sklearn.preprocessing.Normalizer':{
            'norm':['l1', 'l2', 'max']
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
}
