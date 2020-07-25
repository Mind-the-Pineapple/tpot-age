import os
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
from matplotlib.pylab import plt
import argparse

from BayOptPy.tpot.extended_tpot import ExtendedTPOTRegressor
from BayOptPy.tpot.custom_tpot_config_dict import tpot_config_custom
from BayOptPy.tpot.gpr_tpot_config_dict import tpot_config_gpr
from BayOptPy.helperfunctions import get_paths
parser = argparse.ArgumentParser()
parser.add_argument('-nogui',
                    dest='nogui',
                    action='store_true',
                    help='No gui'
                    )
parser.add_argument('-debug',
                    dest='debug',
                    action='store_true',
                    help='Run debug with Pycharm'
                   )
parser.add_argument('-config_dict',
                    dest='config_dict',
                    help='Specify which TPOT config dict to use',
                    choices=['None', 'light', 'custom', 'gpr']
                    )
args = parser.parse_args()

dataset = 'BOSTON'
print('The current dataset being used is: %s' %dataset)
print('The current args are: %s' %args)

# check which TPOT dictionary containing the operators and parameters to be used was passed as argument
if args.config_dict == 'None':
    tpot_config = None
elif args.config_dict == 'light':
    tpot_config = 'TPOT light'
elif args.config_dict == 'custom':
    tpot_config = tpot_config_custom
elif args.config_dict == 'gpr':
    tpot_config = tpot_config_gpr

random_seed = 42
housing = load_boston()
X_train, X_test, y_train, y_test = \
    train_test_split(housing.data, housing.target, train_size=0.75, test_size=0.25, random_state=random_seed)
# used scoring
scoring = 'neg_mean_absolute_error'
cwd = os.getcwd()
best_pipe_paths = os.path.join(cwd, 'BayOptPy/tpot')
# create a directory where to cache the results
tpot = ExtendedTPOTRegressor(generations=5,
                     population_size=50,
                     verbosity=2,
                     random_state=42,
                     config_dict=tpot_config,
                     periodic_checkpoint_folder=best_pipe_paths,
                     scoring=scoring
                     )
tpot.fit(X_train, y_train, X_test, y_test)
print('Test score using optimal model: %f ' %tpot.score(X_test, y_test))

# get paths of where to save the files
project_wd, _, _ = get_paths(args.debug, dataset)
tpot.export(os.path.join(project_wd, 'BayOptPy', 'tpot', 'debug',
                        'tpot_boston_pipeline_super.py'))

# Do some preprocessing to find models where all predictions have the same value and eliminate them, as those will correspond
# to NaN entries or very small numbers on the correlation matrix.
repeated_idx = np.argwhere([np.array_equal(np.repeat(tpot.predictions[i][0], len(tpot.predictions[i])), tpot.predictions[i]) for i in range(len(tpot.predictions))])
print('Index of the models with the same prediction for all subjects: ' + str(np.squeeze(repeated_idx)))
print('Number of models analysed: %d' %len(tpot.predictions))
tpot_predictions = np.delete(np.array(tpot.predictions), np.squeeze(repeated_idx), axis=0)
print('Number of models that will be used for cross-correlation: %s' %(tpot_predictions.shape,))

# Cross correlate the predictions
corr_matrix = np.corrcoef(tpot_predictions)

print('Check the number of NaNs after deleting models with constant predictions: %d' %len(np.argwhere(np.isnan(corr_matrix))))
#colormap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr_matrix, cmap='coolwarm')
plt.title(args.config_dict)
plt.savefig(os.path.join(project_wd, 'BayOptPy', 'tpot', 'cross_corr_%s.png' %args.config_dict))
if not args.nogui:
    plt.show()


from scipy.cluster.hierarchy import dendrogram, linkage
import pdb
methods = ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']
for method in methods:
    plt.figure()
    Z = linkage(corr_matrix, method=method)
    dend = dendrogram(Z, leaf_font_size=8.)
    plt.title(str(method))
    plt.savefig('dendrogram_%s.png' %method)


print('Plot PCA with 95 variance')
from sklearn.decomposition import PCA
pca = PCA(n_components=.95)
corr_matrix_pca = pca.fit_transform(corr_matrix)
plt.scatter(corr_matrix_pca[:, 0], corr_matrix_pca[:, 1])
methods = ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']
for method in methods:
    plt.figure()
    Z = linkage(corr_matrix_pca, method=method)
    dend = dendrogram(Z, leaf_font_size=8.)
    plt.title(str(method))
    plt.savefig('dendrogram_%s_pca.png' %method)

# Once we found the number of clusters perform Agglomartive clustering from sklearn
from sklearn.cluster import AgglomerativeClustering

aggclu = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
clusters_labels = aggclu.fit_predict(corr_matrix)

# plot cluster labeling on the PCA dataset
plt.figure()
plt.scatter(corr_matrix_pca[:, 0], corr_matrix_pca[:, 1], c=clusters_labels, cmap='rainbow')
plt.show()

# check the groupings

# plot using MeanShift
# def plot_clusters(labels, n_clusters, cluster_centers, analysis, corr_matrix):
#     colors = ['#dd4132', '#FF7F50', '#FFA500', '#228B22', '#90EE90', '#40E0D0', '#66CDAA', '#B0E0E6', '#1E90FF']
#     plt.figure()
#     if analysis == 'KMeans' or analysis == 'MeanShift':
#         for k, col in zip(range(n_clusters), colors):
#             my_members = (labels == k)
#             cluster_center = cluster_centers[k]
#             plt.scatter(corr_matrix[my_members, 0], corr_matrix[my_members, 1], c=col, marker='.')
#             plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
#                      markeredgecolor='k', markersize=14)
#             plt.title(analysis)

#     if analysis == 'DBSCAN':
#         core_samples_mask = np.zeros_like(labels, dtype=bool)
#         # Black removed and is used for noise instead.
#         unique_labels = set(labels)
#         colors = [plt.cm.Spectral(each)
#                   for each in np.linspace(0, 1, len(unique_labels))]
#         for k, col in zip(unique_labels, colors):
#             if k == -1:
#                 # Black used for noise.
#                 col = [0, 0, 0, 1]

#             class_member_mask = (labels == k)

#             xy = corr_matrix[class_member_mask & core_samples_mask]
#             plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#                      markeredgecolor='k', markersize=14)

#             xy = corr_matrix[class_member_mask & ~core_samples_mask]
#             plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#                      markeredgecolor='k', markersize=6)
#             plt.title(analysis)

# from sklearn.cluster import MeanShift
# ms = MeanShift()
# ms.fit(corr_matrix)
# labels_unique = np.unique(ms.labels_)
# n_clusters_ = len(labels_unique)
# cluster_centers = ms.cluster_centers_
# labels = ms.labels_
# print('Estimated number of clusters: %d' % n_clusters_)
# plot_clusters(labels, n_clusters_, cluster_centers, 'MeanShift', corr_matrix)

# k-means
# Try using the elbow method
# n_clusters = 4
# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=n_clusters, random_state=42)
# kmeans.fit(corr_matrix)
# labels = kmeans.labels_
# cluster_centers = kmeans.cluster_centers_
# plot_clusters(labels, n_clusters, cluster_centers, 'KMeans', corr_matrix)

# # K-means with pca
# from sklearn.decomposition import PCA
# pca = PCA(n_components=.95)
# corr_matrix_pca = pca.fit_transform(corr_matrix)
# kmeans_pca = KMeans(n_clusters=n_clusters, random_state=42)
# kmeans_pca.fit(corr_matrix_pca)
# labels = kmeans_pca.labels_
# cluster_centers = kmeans_pca.cluster_centers_
# plot_clusters(labels, n_clusters, cluster_centers, 'KMeans', corr_matrix_pca)

# from sklearn.cluster import DBSCAN
# db = DBSCAN(min_samples=3).fit(corr_matrix)
# labels = db.labels_
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# n_noise_ = list(labels).count(-1)
# print('Estimated number of clusters: %d' % n_clusters_)
# print('Estimated number of noise points: %d' % n_noise_)
# plot_clusters(labels, n_clusters, None, 'DBSCAN', corr_matrix)

if not args.nogui:
    plt.show()
