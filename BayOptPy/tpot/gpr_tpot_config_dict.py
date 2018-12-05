# Custom defined list of Gaussian Process regression models to be used by TPOT
import numpy as np
import pdb

# Define list of Kernels
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)

# RBF_kernels = [RBF(length_scale=x, length_scale_bounds=(y, z)) for x, y, z in
#                    zip([1.],
#                        [1e-01],
#                        [10.])]

kernels = [1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)),
           1.0 * RBF(length_scale=.8, length_scale_bounds=(1e-2, 100.0)),
           1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1),
           1.0 * ExpSineSquared(length_scale=1.0, periodicity=3.0,
                length_scale_bounds=(0.1, 10.0), periodicity_bounds=(1.0, 10.0)),
          ]

tpot_config_gpr = {
    'sklearn.gaussian_process.GaussianProcessRegressor': {
        'kernel': kernels,
    },
}
