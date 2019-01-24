import pickle
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-decomposition',
                    dest='decomposition',
                    help='Dimensionality reduction algorithm',
                    choices=['tSNE', 'PCA']
                    )
parser.add_argument('-nclusters',
                    dest='nclusters',
                    help='Number of clusters for k-means',
                    type=int
                    )
args = parser.parse_args()

# read in pickle with the saved results
# with open('BANC_pipelines.pkl', 'rb') as handle:
#     banc_pickle_dump = pickle.load(handle)
banc_pickle_dump = joblib.load('/code/BayOptPy/tpot/tpot_BANC_freesurf_gpr_10gen_.dump')
print('pickle file loaded')

mae_all = []
age_all = []
for key in banc_pickle_dump['evaluated_individuals_'].keys():
    mae_all.append(banc_pickle_dump['evaluated_individuals_'][key]['internal_cv_score'])
print('saved all mae')

# plot results
x = range(len(mae_all))
plt.scatter(x, mae_all)
plt.xlabel('Models')
plt.ylabel('MAE')
plt.savefig('/code/BayOptPy/tpot/MAE_all_models.png')
plt.close()
#plt.show()

# Zoom in into the Y-axis
# plt.scatter(x, mae_all)
# plt.ylim([-10, -5])
#plt.show()

# Get the space with the age prediction
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

age_prediction = np.zeros((len(banc_pickle_dump['predictions']), len(banc_pickle_dump['predictions'][0])))
for test_subject in range(len(banc_pickle_dump['predictions'][0])):
    age_prediction[test_subject, :] = banc_pickle_dump['predictions'][test_subject]

# age_prediction is formatted so that you have model x subject
age_prediction = np.transpose(age_prediction)
print('Data shape: %s' %(age_prediction.shape,))
# standardise age feature before feeding it to the dimensionality reduction
# algorithms. This scales every feature separalty
scaler = StandardScaler()
scaled_age_predictions = scaler.fit_transform(age_prediction)

print('Performing dimensionality reduction: %s' %(args.decomposition))
if args.decomposition == 'tSNE':
    X_embedded = TSNE(n_components=2,
            random_state=0).fit_transform(scaled_age_predictions)
elif args.decomposition == 'PCA':
    pca = PCA(n_components=2, random_state=0)
    X_embedded = pca.fit_transform(scaled_age_predictions)
    print('  Explained variance: ' + ''.join([str(x) + ' ' for x in
        pca.explained_variance_ratio_]))
    print('Find features that mostly contribute to each PC')
    pc0 = max(pca.components_[0], key=abs)
    pc1 = max(pca.components_[1], key=abs)
    idx_model_pc0 = np.where(pca.components_ == pc0)[1][0]
    idx_model_pc1 = np.where(pca.components_ == pc1)[1][0]
    print('The most influential model for PC0:')
    print(banc_pickle_dump['pipelines'][idx_model_pc0])
    print('The most influential model for PC1:')
    print(banc_pickle_dump['pipelines'][idx_model_pc1])

# Plot the embedded space
plt.figure()
plt.scatter(X_embedded[:, 0], X_embedded[:, 1])
plt.title('%s' %(args.decomposition))
plt.close()

# Do very simple k-means to find the clusters and the belonging of each model
colours = ['b', 'o', 'g']
kmeans = KMeans(n_clusters=args.nclusters, random_state=0).fit(X_embedded)
# plot the embedded space with the kmeans
plt.figure()
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=kmeans.labels_)
plt.title('10 gen')
plt.savefig('/code/BayOptPy/tpot/%s_analysis/%s_10gen_42seed_gpr.png'
        %(args.decomposition, args.decomposition))
plt.close()

print('Number of elements in each cluster')
unique, counts = np.unique(kmeans.labels_, return_counts=True)
print(dict(zip(unique, counts)))

#check predictions of the clusters with lower density
# transform predictions into a numpy array
predictions = np.array(banc_pickle_dump['predictions'])
# Calculate the difference between the predictions and the true age, for every
with open('/code/BayOptPy/tpot/age_traindata.pickle', 'rb') as handle:
    Ytest = pickle.load(handle)
corrected_predictions = predictions - Ytest
corrected_predictions = np.transpose(corrected_predictions)
predictions = np.transpose(predictions)
cluster_predictions = {}
cluster_corrected = {}
for cluster in range(args.nclusters):
    cluster_predictions[cluster] = predictions[kmeans.labels_ == cluster]
    cluster_corrected[cluster] = corrected_predictions[kmeans.labels_ == cluster]

# plot the distances to get a feeling for the data. Note: For the following
# plots every colour represents a different model.
for cluster in range(args.nclusters):
    plt.figure()
    for element in range(cluster_corrected[cluster].shape[0]):
        plt.scatter(range(len(predictions[kmeans.labels_ == 0][0])),
                cluster_corrected[cluster][element])
    plt.xlabel('Analysed Models')
    plt.ylabel('Predicted Age - True Age')
    plt.title('%s' %args.decomposition)
    plt.savefig('/code/BayOptPy/tpot/%s_analysis/age_overview_cluster%d.png'
            %(args.decomposition, cluster))
    plt.close()

# Get the correlation between the age in the different cluster and true age
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

results = {}
for cluster in range(args.nclusters):
    print('Analysis cluster #%d' %cluster)
    results[cluster] = {}
    results[cluster]['correlation'] = []
    results[cluster]['mae'] = []

    for model in range(cluster_predictions[cluster].shape[1]):
        corr, _ = pearsonr(cluster_predictions[cluster][:, model],
                Ytest[kmeans.labels_ == cluster])
        results[cluster]['correlation'].append(corr)
        mae = mean_absolute_error(Ytest[kmeans.labels_ == cluster],
                cluster_predictions[cluster][:, model])
        results[cluster]['mae'].append(mae)

# plot the correlations for the different clusters
plt.figure()
for cluster in range(args.nclusters):
    plt.scatter(range(predictions[kmeans.labels_ == cluster].shape[1]),
            results[cluster]['correlation'], label='cluster%d' %cluster)

plt.legend()
plt.xlabel('Analysed Models')
plt.ylabel('Correlation')
plt.title('%s' %args.decomposition)
plt.savefig('/code/BayOptPy/tpot/%s_analysis/correlations.png'
        %args.decomposition)
plt.close()

 # get the mae. Note: A low MAE means good prediction acction. MAE represents
 # actual value - predicted valeu
plt.figure()
for cluster in range(args.nclusters):
    plt.scatter(range(predictions[kmeans.labels_ == cluster].shape[1]),
    results[cluster]['mae'], label='cluster%d' %cluster)
plt.legend()
plt.xlabel('Analysed Models')
plt.title('%s' %args.decomposition)
plt.ylabel('MAE')
plt.savefig('/code/BayOptPy/tpot/%s_analysis/mae.png' %args.decomposition)
plt.close()
