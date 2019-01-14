import pickle
import matplotlib.pyplot as plt

# read in pickle with the saved results
with open('BANC_pipelines.pkl', 'rb') as handle:
    banc_pickle_dump = pickle.load(handle)
print('pickle file loaded')

mae_all = []
age_all = []
for key in banc_pickle_dump.keys():
    mae_all.append(banc_pickle_dump[key]['internal_cv_score'])
print('saved all mae')

# plot results
x = range(len(mae_all))
plt.scatter(x, mae_all)
plt.xlabel('Models')
plt.ylabel('MAE')
plt.show()

