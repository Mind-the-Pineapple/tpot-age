from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
from matplotlib.pylab import plt
import argparse

from BayOptPy.tpot.extended_tpot import ExtendedTPOTRegressor

parser = argparse.ArgumentParser()
parser.add_argument('-nogui',
                    dest='nogui',
                    action='store_true',
                    help='No gui'
                    )
args = parser.parse_args()

random_seed = 42
housing = load_boston()
X_train, X_test, y_train, y_test = \
    train_test_split(housing.data, housing.target, train_size=0.75, test_size=0.25, random_state=random_seed)
# used scoring
scoring = 'neg_mean_absolute_error'
tpot_config = {
    'sklearn.linear_model.ElasticNetCV': {
    'l1_ratio': np.arange(0.0, 1.01),
    'tol': [1e-5]
    },
    'sklearn.neighbors.KNeighborsRegressor': {
    'n_neighbors': range(1,2),
    'weights': ["uniform", "distance"],
    'p': [1, 2]
    },
    # preprocessing
    'sklearn.decomposition.PCA': {
    'svd_solver': ['randomized'],
    'iterated_power': range(1,2)
    }
}
# create a directory where to cache the results
tpot = ExtendedTPOTRegressor(generations=5,
                     population_size=50,
                     verbosity=2,
                     random_state=42,
                     config_dict='TPOT light',
                     scoring=scoring
                     )
tpot.fit(X_train, y_train, X_test)
print('Test score using optimal model: %f ' %tpot.score(X_test, y_test))
tpot.export('BayOptPy/tpot/debug/tpot_boston_pipeline_super.py')

# Cross correlate the predictions
corr_matrix = np.corrcoef(tpot.predictions)
#colormap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr_matrix, cmap='coolwarm')
plt.savefig('cross_corr.png')


# plot using MeanShift
def plot_clusters(labels, n_clusters, cluster_centers, analysis, corr_matrix):
    colors = ['#dd4132', '#FF7F50', '#FFA500', '#228B22', '#90EE90', '#40E0D0', '#66CDAA', '#B0E0E6', '#1E90FF']
    plt.figure()
    if analysis == 'KMeans' or analysis == 'MeanShift':
        for k, col in zip(range(n_clusters), colors):
            my_members = (labels == k)
            cluster_center = cluster_centers[k]
            plt.scatter(corr_matrix[my_members, 0], corr_matrix[my_members, 1], c=col, marker='.')
            plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                     markeredgecolor='k', markersize=14)
            plt.title(analysis)

    if analysis == 'DBSCAN':
        core_samples_mask = np.zeros_like(labels, dtype=bool)
        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (labels == k)

            xy = corr_matrix[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=14)

            xy = corr_matrix[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=6)
            plt.title(analysis)

from sklearn.cluster import MeanShift
ms = MeanShift()
ms.fit(corr_matrix)
labels_unique = np.unique(ms.labels_)
n_clusters_ = len(labels_unique)
cluster_centers = ms.cluster_centers_
labels = ms.labels_
print('Estimated number of clusters: %d' % n_clusters_)
plot_clusters(labels, n_clusters_, cluster_centers, 'MeanShift', corr_matrix)

# k-means
# Try using the elbow method
n_clusters = 4
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(corr_matrix)
labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_
plot_clusters(labels, n_clusters, cluster_centers, 'KMeans', corr_matrix)

# K-means with pca
from sklearn.decomposition import PCA
pca = PCA(n_components=.95)
corr_matrix_pca = pca.fit_transform(corr_matrix)
kmeans_pca = KMeans(n_clusters=n_clusters, random_state=42)
kmeans_pca.fit(corr_matrix_pca)
labels = kmeans_pca.labels_
cluster_centers = kmeans_pca.cluster_centers_
plot_clusters(labels, n_clusters, cluster_centers, 'KMeans', corr_matrix_pca)

from sklearn.cluster import DBSCAN
db = DBSCAN(min_samples=3).fit(corr_matrix)
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
plot_clusters(labels, n_clusters, None, 'DBSCAN', corr_matrix)

if not args.nogui:
    plt.show()
