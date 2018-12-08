import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import (cross_val_predict, train_test_split,
                                    cross_validate)
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.gaussian_process import GaussianProcessRegressor


import pdb
from BayOptPy.helperfunctions import get_data, get_paths
from BayOptPy.sklearn.myLinearKernel import LinearKernel

# Perform a simple analysis using the GPR with linear kernel and analyse the
# score

# Divide the data into training and test set
random_seed = 42
debug = True
# dataset = 'BANC_freesurf'
dataset = 'BANC_freesurf'
resamplefactor = 4

project_wd, project_data, project_sink = get_paths(debug, dataset)
demographics, imgs, data = get_data(project_data, dataset, debug, project_wd,
                                    resamplefactor)

targetAttribute = np.array(demographics['Age'])
# To ensure the example runs quickly, we'll make the training dataset relatively small
Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, targetAttribute, test_size=.25,
                                                                random_state=random_seed)

# Perform analysis with custom Linear Kernel
kernel1 = LinearKernel()
gp1 = GaussianProcessRegressor(kernel=kernel1, normalize_y=True)
# gp.fit(Xtrain, Ytrain)
# Ypredict = gp.predict(Xtest)

cv_results1 = cross_validate(gp1, Xtrain, Ytrain,
                            scoring='neg_mean_absolute_error', cv=10)
print('The mean absolute error over the different cross-validations is:')
print(np.mean(cv_results1['test_score']))

# Perform analysis with dot product kernel and set the sigma to 0
kernel2 = DotProduct(sigma_0=0)
gp2 = GaussianProcessRegressor(kernel=kernel2, normalize_y=True)
cv_results2 = cross_validate(gp2, Xtrain, Ytrain,
                            scoring='neg_mean_absolute_error', cv=10)
print('The mean absolute error over the different cross-validations is:')
print(np.mean(cv_results2['test_score']))

# Plot prediction os two kernels
Ypredict1 = cross_val_predict(gp1, Xtrain, Ytrain, cv=10)
Ypredict2 = cross_val_predict(gp2, Xtrain, Ytrain, cv=10)
# plot predicted vs true
plt.figure()
pdb.set_trace()
plt.scatter(Ytrain, Ypredict1, c='b')
plt.scatter(Ytrain, Ypredict2, c='r')
plt.show()

