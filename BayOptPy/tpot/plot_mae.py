import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import pandas as pd

# OASIS with different numbers of Generations (results are saved here: qsubBayOpt_OASIS.job.o4326307)
# cv=5, dask=False, dataset='OASIS', debug=False, generations=30, nogui=False, population_size=100
dic_gen = {'MAE': [-10.157332265800637, -10.13726144682216, -10.13726144682216, -10.13726144682216, -10.049021689766189,
                   -10.048997408372596, -10.048997408372596, -10.048997408372596, -9.66793351168611, -9.66793351168611,
                   -9.66793351168611, -9.66793351168611, -9.66793351168611, -9.667933511686096, -9.667933511686094,
                   -9.588041303518294, -9.588041303518294, -9.537413594875314, -9.230141170150933, -9.230141170150933,
                   -9.230141170150933, -9.230141170150933, -9.230141170150933, -9.230141170150933, -9.062720805440987,
                   -9.062720805440987, -9.062720805440987, -9.062720805440987, -9.062720805440987, -9.062720805440987],
           'n_generations': np.arange(1, 30 + 1),
           }


# OASIS for 60 generations (results are saved here: qsubBayOpt_OASIS.job.o4326309)
# Namespace(cv=5, dask=False, dataset='OASIS', debug=False, generations=60, nogui=False, population_size=100)
#
dic_gen2 = {'MAE': [-10.157332265800644, -10.148634423480523, -10.148634423480523, -10.027366493285328, -10.027366493285328,
                    -9.230141170150954, -9.230141170150954, -9.230141170150954, -9.230141170150954, -9.230141170150954,
                    -9.230141170150954, -9.230141170150954, -9.230141170150954, -9.079230477225071, -9.079230477225071,
                    -9.079230477225071, -9.079230477225071, -9.00487985132841, -9.00487985132841, -9.00487985132841,
                    -9.00487985132841, -8.94430617608224, -8.94430617608224, -8.94430617608224, -8.94430617608224,
                    -8.94430617608224, -8.927656932206142, -8.927656932206142, -8.927656932206103, -8.922154327148041,
                    -8.877289485646749, -8.877289485646749, -8.877289485646749, -8.877289485646749, -8.855758301907134,
                    -8.855758301907134, -8.758534150239667, -8.685490337016068, -8.685490337016068, -8.685490337016068,
                    -8.685490337016068, -8.685490337016068, -8.685490337016068, -8.685490337016068, -8.685490337016068,
                    -8.685490337016068, -8.685490337016068, -8.6225812452792, -8.6225812452792, -8.6225812452792,
                    -8.6225812452792, -8.6225812452792, -8.6225812452792, -8.6225812452792, -8.6225812452792,
                    -8.6225812452792, -8.6225812452792, -8.6225812452792, -8.6225812452792, -8.6225812452792
                    ],
            'n_generations': np.arange(1, 60+1)}
# OASIS with different numbers of resampling factors
# (cv=5, dask=False, dataset='OASIS', debug=False, generations=5, nogui=False,
# population_size=100)
dic_res = {'MAE': [
              # with preprocessing
              -7.2159810126582276, -7.531252912387068,-8.942871304902122, -7.549231447189402, -7.772747987489725,
              -8.457877460092652, -9.4620802412677, -8.09883877634299, -8.080819808555542, -8.049551694255202,
              # no preprocessing
              -8.179169010196004, -7.708518645227505, -7.787925092141313, -7.3851717902350815, -7.7594936708860756,
              -7.371039860280367, -7.711408527605223, -7.749464796484447, -9.26143546764901, -7.915995397008055
                   ],
          'resample_factor': list(np.arange(1,10+1)) + list(np.arange(1, 10+1)),
          'Pipeline':[
          'KNeighborsRegressor(RidgeCV(input_matrix), n_neighbors=8, p=2,        \
                       weights=uniform)',
          'KNeighborsRegressor(RidgeCV(input_matrix), n_neighbors=27, p=2,        \
                       weights=distance)',
          'LinearSVR(KNeighborsRegressor(input_matrix, n_neighbors=21, p=1,      \
                       weights=uniform), C=1.0, dual=True, epsilon=0.01,          \
                       loss=epsilon_insensitive, tol=1e-05)',
          'KNeighborsRegressor(RidgeCV(KNeighborsRegressor(MaxAbsScaler(input_matrix), \
                       n_neighbors=36, p=1, weights=uniform)), n_neighbors=27, p=2, \
                       weights=distance)',
          'KNeighborsRegressor(RidgeCV(SelectFwe(input_matrix, alpha=0.046)), \
                        n_neighbors=9, p=2, weights=distance)',
          'DecisionTreeRegressor(ElasticNetCV(MaxAbsScaler(input_matrix),         \
                        l1_ratio=0.05, tol=0.01), max_depth=5,                    \
                        min_samples_leaf=10, min_samples_split=14)',
          'RidgeCV(RobustScaler(DecisionTreeRegressor(input_matrix,               \
                        max_depth=6, min_samples_leaf=12, min_samples_split=6)))',
          'DecisionTreeRegressor(ElasticNetCV(MinMaxScaler(input_matrix),         \
                        l1_ratio=0.05, tol=0.0001), max_depth=5, min_samples_leaf=10, \
                        min_samples_split=11)',
          'DecisionTreeRegressor(ElasticNetCV(input_matrix, l1_ratio=0.05, tol=0.0001), \
                        max_depth=2, min_samples_leaf=10,  min_samples_split=11)',
          'KNeighborsRegressor(RidgeCV(input_matrix), n_neighbors=27, p=2, weights=distance)',
          # without preprocessing
          'RidgeCV(KNeighborsRegressor(RidgeCV(input_matrix), n_neighbors=23, p=2, weights=distance))',
          'KNeighborsRegressor(LassoLarsCV(RidgeCV(input_matrix), normalize=True), n_neighbors=37, p=2, weights=uniform)',
          'KNeighborsRegressor(RidgeCV(VarianceThreshold(input_matrix, threshold=0.0005)), n_neighbors=22, p=2, weights=distance)',
          'KNeighborsRegressor(RidgeCV(RidgeCV(SelectPercentile(input_matrix, percentile=65))), n_neighbors=7, p=1, weights=uniform)',
          'KNeighborsRegressor(RidgeCV(input_matrix), n_neighbors=3, p=1, weights=uniform)',
          'DecisionTreeRegressor(SelectFwe(ElasticNetCV(input_matrix, l1_ratio=0.25, tol=0.01), alpha=0.037), max_depth=5, min_samples_leaf=9, min_samples_split=4)',
          'ElasticNetCV(KNeighborsRegressor(RidgeCV(input_matrix), n_neighbors=13, p=1, weights=distance), l1_ratio=0.9, tol=0.001)',
          'LassoLarsCV(KNeighborsRegressor(ElasticNetCV(input_matrix, l1_ratio=0.05, tol=0.0001), n_neighbors=13, p=2, weights=uniform), normalize=False)',
          'KNeighborsRegressor(LinearSVR(input_matrix, C=0.01, dual=False, epsilon=0.1, loss=squared_epsilon_insensitive, tol=1e-05), n_neighbors=3, p=1, weights=distance)',
          'Best pipeline: KNeighborsRegressor(RidgeCV(input_matrix), n_neighbors=22, p=1, weights=uniform)'
           ],
           'Preprocessing': ['with_preprocessing', 'with_preprocessing', 'with_preprocessing', 'with_preprocessing',
                             'with_preprocessing', 'with_preprocessing', 'with_preprocessing', 'with_preprocessing',
                             'with_preprocessing', 'with_preprocessing',
                             'no_preprocessing', 'no_preprocessing', 'no_preprocessing', 'no_preprocessing',
                             'no_preprocessing', 'no_preprocessing', 'no_preprocessing', 'no_preprocessing',
                             'no_preprocessing', 'no_preprocessing']
     }

# Analyse the effect of different resampling
print('Plot Resampling Analysis')
df_res = pd.DataFrame.from_dict(dic_res)
plt.figure()
#sns.scatterplot(x='resample_factor', y='MAE', data=df_res, hue='Preprocessing')
sns.lineplot(x='resample_factor', y='MAE', markers=['o', 'o'], dashes=False, style='Preprocessing', hue='Preprocessing', data=df_res)
plt.title('Resampling factor')
plt.savefig('MAE_preproc.png')

# Analyse the progression of MAE with the number of generations
print('Plot Analysis of the number of generations')
df_gen = pd.DataFrame.from_dict(dic_gen)
plt.figure()
sns.lineplot(x='n_generations', y='MAE', markers='o', data=df_gen)
sns.scatterplot(x='n_generations', y='MAE', data=df_gen)
plt.savefig('MAE_gen.png')

# Analyse the progression of MAE with the number of generations
print('Plot Analysis of the number of generations')
df_gen2 = pd.DataFrame.from_dict(dic_gen2)
plt.figure()
sns.lineplot(x='n_generations', y='MAE', markers='o', data=df_gen2)
sns.scatterplot(x='n_generations', y='MAE', data=df_gen2)
plt.savefig('MAE_gen2.png')
