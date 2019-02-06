# Custom defined list of Gaussian Process regression models to be used by TPOT
import numpy as np
import pdb
from itertools import product

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
        'alpha': [1e-10, 1e-8, 1e-5, 1e-3, 1e-2, 1e0]
    },
}
