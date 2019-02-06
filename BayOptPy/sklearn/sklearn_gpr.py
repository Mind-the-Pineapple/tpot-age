import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct

from BayOptPy.helperfunctions import get_data, get_paths

debug = False
dataset = 'BANC'
resamplefactor = 1

random_seed = 42

project_wd, project_data, project_sink = get_paths(debug, dataset)
demographics, imgs, data = get_data(project_data, dataset, debug, project_wd, resamplefactor)

# Get the fsl data, concatenate GM and WM. For a start use only the WM
targetAttribute = np.array(demographics['Age'])

# Train the model
kernel = DotProduct(sigma_0=0)
gp2 = GaussianProcessRegressor(kernel=kernel, normalize_y=False)
cv_results2 = cross_validate(gp2, data, targetAttribute,
                            scoring='neg_mean_absolute_error', cv=10, n_jobs=4)
# Do cross-validation
print('The MAE are:')
[print(i) for i in cv_results2['test_score']]
print('The mean absolute error over the different cross-validations is:')
print(np.mean(cv_results2['test_score']))
