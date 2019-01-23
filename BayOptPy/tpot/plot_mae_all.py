import pickle
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
#plt.show()

# Zoom in into the Y-axis
# plt.scatter(x, mae_all)
# plt.ylim([-10, -5])
#plt.show()

# Get the space with the age prediction
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

age_prediction = np.zeros((len(banc_pickle_dump['predictions']), len(banc_pickle_dump['predictions'][0])))
for test_subject in range(len(banc_pickle_dump['predictions'][0])):
    age_prediction[test_subject, :] = banc_pickle_dump['predictions'][test_subject]

X_embedded = TSNE(n_components=2, random_state=0).fit_transform(age_prediction)
plt.scatter(X_embedded[:, 0], X_embedded[:, 1])

# Do very simple k-means to find the clusters and the belonging of each model
kmeans = KMeans(n_clusters=3, random_state=0).fit(X_embedded)
# plot the embedded space with the kmeans
plt.figure()
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=kmeans.labels_)
plt.title('10 gen')
plt.savefig('/code/BayOptPy/tpot/tSNE_analysis/tSNE_10gen_42seed_gpr.png')

print('Number of elements in each cluster')
unique, counts = np.unique(kmeans.labels_, return_counts=True)
print(dict(zip(unique, counts)))

#check predictions of the clusters with lower density
# transform predictions into a numpy array
predictions = np.array(banc_pickle_dump['predictions'])
cluster0 = predictions[kmeans.labels_ == 0]
cluster1 = predictions[kmeans.labels_ == 1]
cluster2 = predictions[kmeans.labels_ == 2]

# Calculate the difference between the predictions and the true age, for every
with open('age_traindata.pickle', 'rb') as handle:
    Ytest = pickle.load(handle)
# subtract the original age to the predicted age
cluster0_corrected = cluster0 - Ytest
cluster1_corrected = cluster1 - Ytest
cluster2_corrected = cluster2 - Ytest

# plot the distances to get a feeling for the data. Note: For the following
# plots every colour represents a different model.
plt.figure()
for element in range(cluster0_corrected.shape[0]):
    plt.scatter(range(len(predictions[kmeans.labels_ == 0][0])),
            cluster0_corrected[element])
plt.xlabel('Analysed Subjects')
plt.ylabel('Predicted Age - True Age')
plt.savefig('/code/BayOptPy/tpot/tSNE_analysis/age_overview_cluster0.png')

plt.figure()
for element in range(cluster1_corrected.shape[0]):
    plt.scatter(range(len(predictions[kmeans.labels_ == 1][0])),
            cluster1_corrected[element])
plt.xlabel('Analysed Subjects')
plt.ylabel('Predicted Age - True Age')
plt.savefig('/code/BayOptPy/tpot/tSNE_analysis/age_overview_cluster1.png')

plt.figure()
for element in range(cluster2_corrected.shape[0]):
    plt.scatter(range(len(predictions[kmeans.labels_ == 2][0])),
            cluster2_corrected[element])
plt.xlabel('Analysed Subjects')
plt.ylabel('Predicted Age - True Age')
plt.savefig('/code/BayOptPy/tpot/tSNE_analysis/age_overview_cluster2.png')

# Get the correlation between the age in the different cluster and true age
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

cluster0_correlation = []
cluster0_mae = []
for model in range(cluster0.shape[0]):
    corr, _ = pearsonr(cluster0[model], Ytest)
    cluster0_correlation.append(corr)
    mae = mean_absolute_error(Ytest, cluster0[model])
    cluster0_mae.append(mae)

cluster1_correlation = []
cluster1_mae = []
for model in range(cluster1.shape[0]):
    corr, _ = pearsonr(cluster1[model], Ytest)
    cluster1_correlation.append(corr)
    mae = mean_absolute_error(Ytest, cluster1[model])
    cluster1_mae.append(mae)

cluster2_correlation = []
cluster2_mae = []
for model in range(cluster2.shape[0]):
    corr, _ = pearsonr(cluster2[model], Ytest)
    cluster2_correlation.append(corr)
    mae = mean_absolute_error(Ytest, cluster2[model])
    cluster2_mae.append(mae)

# plot the correlations for the different clusters
plt.figure()
plt.scatter(range(predictions[kmeans.labels_ == 0].shape[0]), cluster0_correlation,
        label='cluster0')
plt.scatter(range(predictions[kmeans.labels_ == 1].shape[0]), cluster1_correlation,
        label='cluster1')
plt.scatter(range(predictions[kmeans.labels_ == 2].shape[0]), cluster2_correlation,
        label='cluster2')
plt.legend()
plt.xlabel('Analysed Subjects')
plt.ylabel('Correlation')
plt.savefig('/code/BayOptPy/tpot/tSNE_analysis/correlations.png')

 # get the mae. Note: A low MAE means good prediction acction. MAE represents
 # actual value - predicted valeu
plt.figure()
plt.scatter(range(predictions[kmeans.labels_ == 0].shape[0]), cluster0_mae,
        label='cluster0')
plt.scatter(range(predictions[kmeans.labels_ == 1].shape[0]), cluster1_mae,
        label='cluster1')
plt.scatter(range(predictions[kmeans.labels_ == 2].shape[0]), cluster2_mae,
        label='cluster2')
plt.legend()
plt.xlabel('Analysed Subjects')
plt.ylabel('MAE')
plt.savefig('/code/BayOptPy/tpot/tSNE_analysis/mae.png')

