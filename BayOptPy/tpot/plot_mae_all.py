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
plt.scatter(x, mae_all)
plt.ylim([-10, -5])
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
plt.savefig('/code/BayOptPy/tpot/tSNE_10gen_42seed_gpr.png')

print('Number of elements in each cluster')
unique, counts = np.unique(kmeans.labels_, return_counts=True)
print(dict(zip(unique, counts)))


