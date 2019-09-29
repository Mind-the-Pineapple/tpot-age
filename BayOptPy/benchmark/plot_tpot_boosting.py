import os
import pickle

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, friedmanchisquare


from BayOptPy.helperfunctions import (set_publication_style,
                                      plot_confusion_matrix_boosting,
                                      ttest_ind_corrected)

parser = argparse.ArgumentParser()
parser.add_argument('-model',
                    dest='model',
                    help='Define if a classification or regression problem',
                    choices=['regression', 'classification', 'classification2']
                    )
parser.add_argument('-generations',
                     dest='generations',
                     help='Specify number of generations to use',
                     type=int,
                     required=True
                     )
parser.add_argument('-analysis',
                    dest='analysis',
                    help='Specify which type of analysis to use',
                    choices=['vanilla_combi',
                             'uniform_dist',
                             'preprocessing',
                             'population'],
                    required=True
                    )
args = parser.parse_args()

def barplot_annotate_brackets(num1, num2, data, center, height, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None):
    """
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    plt.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    plt.text(*mid, text, **kwargs)


# Settings
#----------------------------------------------------------------------------
set_publication_style()
classes = np.array(['young', 'old', 'adult'], dtype='U10')

if (args.model == 'regression'):
    if args.analysis == 'preprocessing':
        print('Pre-processing analysis')
        preprocessing_types = [
                               'vanilla',
                               'feat_selec',
                               'feat_combi',
                               'vanilla_combi']
        ind = np.arange(0, len(preprocessing_types))
        df = pd.DataFrame(columns=['mae_test', 'r_test', 'preprocessing',
                                   ])
        for preprocessing in preprocessing_types:
            save_path = '/code/BayOptPy/tpot_%s/Output/%s/age/%03d_generations' \
                        %(args.model, preprocessing, args.generations)
            with open(os.path.join(save_path, 'tpot_all_seeds.pckl'), 'rb') as handle:
                tpot_results = pickle.load(handle)
                tpot_results['preprocessing'] = preprocessing
                tpot_results['mean_flatten'] = np.ndarray.flatten(tpot_results['mae_test'])
                # save information to dataframe
                df = df.append(tpot_results, ignore_index=True)

        # Calculate mean for every
        # Plot MAE
        plt.figure(figsize=(10,15))
        plt.bar(ind,
                [np.mean(df['mean_flatten'][0]),
                 np.mean(df['mean_flatten'][1]),
                 np.mean(df['mean_flatten'][2]),
                 np.mean(df['mean_flatten'][3])
                ],
                yerr=[np.std(df['mean_flatten'][0]),
                 np.std(df['mean_flatten'][1]),
                 np.std(df['mean_flatten'][2]),
                 np.mean(df['mean_flatten'][3])
                     ],
                color=['b', 'r', 'g', 'orange']
                     )
        plt.xticks(ind, (preprocessing_types))
        plt.ylim([4, 5])
        plt.yticks(np.arange(4, 5, .2))
        plt.ylabel('MAE')
        plt.savefig(os.path.join(save_path, 'MAE_preprocessinge.eps'))

        data = [df['mean_flatten'][0], df['mean_flatten'][1],
                df['mean_flatten'][2], df['mean_flatten'][3]]

        plt.figure()
        sns.swarmplot(data=data)
        plt.ylabel('MAE')
        plt.xticks(ind, (preprocessing_types))
        plt.savefig(os.path.join(save_path, 'MAE_preprocessing_box.eps'))

        # Print statistics
        f, p = friedmanchisquare(df['mean_flatten'][0], df['mean_flatten'][1],
                                 df['mean_flatten'][2])
        print('Statisitcs')
        print('F-value %.3f' %f)
        print('p-value: %.3f' %p)

        print('Try Bengio Test')
        t, p_t = ttest_ind_corrected(df['mae_test'][1], df['mae_test'][2], k=10,
                                    r=10)
        print('T: %.3f and p: %.f' %(t, p_t))
        t, p_t = ttest_ind_corrected(df['mae_test'][0], df['mae_test'][2], k=10,
                                    r=10)
        print('T: %.3f and p: %.f' %(t, p_t))
        t, p_t = ttest_ind_corrected(df['mae_test'][0], df['mae_test'][1], k=10,
                                    r=10)
        print('T: %.3f and p: %.f' %(t, p_t))

    elif args.analysis == 'population':
        print('Population analysis')
        preprocessing_types = [
                               '00010',
                               '00100',
                               '01000']
        ind = np.arange(0, len(preprocessing_types))
        df = pd.DataFrame(columns=['mae_test', 'r_test', 'preprocessing',
                                   ])
        for preprocessing in preprocessing_types:
            save_path = '/code/BayOptPy/tpot_%s/Output/%s/age/%s_population_size/%03d_generations' \
                        %(args.model, args.analysis, preprocessing, args.generations)
            with open(os.path.join(save_path, 'tpot_all_seeds.pckl'), 'rb') as handle:
                tpot_results = pickle.load(handle)
                tpot_results['preprocessing'] = preprocessing
                tpot_results['mean_flatten'] = np.ndarray.flatten(tpot_results['mae_test'])
                # save information to dataframe
                df = df.append(tpot_results, ignore_index=True)

        # Calculate mean for every
        # Plot MAE
        plt.figure(figsize=(10,15))
        plt.bar(ind,
                [np.mean(df['mean_flatten'][0]),
                 np.mean(df['mean_flatten'][1]),
                 np.mean(df['mean_flatten'][2])],
                yerr=[np.std(df['mean_flatten'][0]),
                 np.std(df['mean_flatten'][1]),
                 np.std(df['mean_flatten'][2]),
                     ],
                color=['b', 'r', 'g']
                     )
        plt.xticks(ind, (preprocessing_types))
        plt.ylim([4, 5])
        plt.yticks(np.arange(4, 5, .2))
        plt.ylabel('MAE')
        plt.savefig(os.path.join(save_path, 'MAE_preprocessinge.eps'))

        data = [df['mean_flatten'][0], df['mean_flatten'][1],
                df['mean_flatten'][2]]

        plt.figure()
        sns.swarmplot(data=data)
        plt.ylabel('MAE')
        plt.yticks(np.arange(4.3, 4.9, .1))
        plt.xticks(ind, (preprocessing_types))
        plt.savefig(os.path.join(save_path, 'MAE_preprocessing_box.eps'))

        # Print statistics
        f, p = friedmanchisquare(df['mean_flatten'][0], df['mean_flatten'][1],
                                 df['mean_flatten'][2])
        print('Statisitcs')
        print('F-value %.3f' %f)
        print('p-value: %.3f' %p)

        print('Try Bengio Test')
        t, p_t = ttest_ind_corrected(df['mae_test'][1], df['mae_test'][2], k=10,
                                    r=10)
        print('T: %.3f and p: %.f' %(t, p_t))
        t, p_t = ttest_ind_corrected(df['mae_test'][0], df['mae_test'][2], k=10,
                                    r=10)
        print('T: %.3f and p: %.f' %(t, p_t))
        t, p_t = ttest_ind_corrected(df['mae_test'][0], df['mae_test'][1], k=10,
                                    r=10)
        print('T: %.3f and p: %.f' %(t, p_t))

    else:
        # Load the dat from the saved pickle
        save_path = '/code/BayOptPy/tpot_%s/Output/%s/age/%03d_generations' \
        %(args.model,args.analysis, args.generations)
        with open(os.path.join(save_path, 'tpot_all_seeds.pckl'), 'rb') as handle:
            tpot_results = pickle.load(handle)

        with open(os.path.join(save_path, 'rvr_all_seeds.pckl'), 'rb') as handle:
            rvr_results = pickle.load(handle)

        # MAE - Validation plot
        #----------------------------------------------------------------------------
        # Do some statistics to see if the results from tpot is significantly differen from rvr
        print('Test dataset')
        print('-------------------------------------------------------------------')
        print('MAE analysis')
        ind = np.arange(2)
        t, prob = ttest_ind_corrected(tpot_results['mae_test'][:10],
                                      rvr_results['mae_test'][:10], k=10, r=10)

        # Test how it would be with the standat TPOT
        seed_tpot_flatten = np.ndarray.flatten(tpot_results['mae_test'])
        seed_rvr_flatten = np.ndarray.flatten(rvr_results['mae_test'])

        t_old, prob_old = ttest_ind(seed_tpot_flatten, seed_rvr_flatten)
        print('T old method')
        print('T-statistics: %.3f, p-value: %.10f' %(t_old, prob_old))

        print('Mean over the different seeds')
        print('Mean %.3f Std %.5f MAE Test TPOT'  %(np.mean(tpot_results['mae_test']),
                                                  np.std(tpot_results['mae_test'])))
        print('Mean %.3f Std %.5f MAE Test RVR' %(np.mean(rvr_results['mae_test']),
                                                 np.std(rvr_results['mae_test'])))
        print('T-statistics: %.3f, p-value: %.10f' %(t, prob))

        plt.figure(figsize=(10,15))
        plt.bar(ind,
                [np.mean(tpot_results['mae_test']), np.mean(rvr_results['mae_test'])],
                yerr=[np.std(tpot_results['mae_test']),
                      np.std(tpot_results['mae_test'])],
                color=['b', 'r']
                     )
        barplot_annotate_brackets(0, 1, '**', ind,
                                  height=[np.mean(tpot_results['mae_test']),
                                          np.mean(rvr_results['mae_test'])])
        plt.xticks(ind, ('TPOT', 'RVR'))
        plt.ylim([4.5, 7])
        plt.yticks(np.arange(4.5, 7.5, .5))
        plt.ylabel('MAE')
        plt.savefig(os.path.join(save_path, 'MAE_bootstrap_test.eps'))
        plt.close()


        # Pearsons Correlation Analysis
        #----------------------------------------------------------------------------
        # Pearsons Correlation - test plot
        print('Pearsons Correlation: Test dataset')
        # t, prob = ttest_ind(tpot_results['r_test'], rvr_results['r_test'])
        t, prob = ttest_ind_corrected(tpot_results['r_test'][:10],
                                      rvr_results['r_test'][:10],
                                     k=10, r=10)
        print('T-statistics: %.3f, p-value: %.25f' %(t, prob))
        print('Mean %.3f Std %.5f Pearsons TPOT' %(np.mean(tpot_results['r_test']),
                                          np.std(tpot_results['r_test'])))
        print('Mean %.3f Std %.5f Pearsons RVR' %(np.mean(rvr_results['r_test']),
                                          np.std(rvr_results['r_test'])))
        plt.figure(figsize=(10,15))
        plt.bar(ind,
                [np.mean(tpot_results['r_test']),
                 np.mean(rvr_results['r_test'])],
                yerr=[np.std(tpot_results['r_test']),
                      np.std(tpot_results['r_test'])],
                color=['b', 'r']
                     )
        plt.xticks(ind, ('TPOT', 'RVR'))
        plt.ylim([.75, 1])
        plt.yticks(np.arange(.75, 1.005, .05))
        barplot_annotate_brackets(0, 1, 'p<.001', ind,
                                  height=[np.mean(tpot_results['r_test']),
                                          np.mean(rvr_results['r_test'])])
        plt.ylabel('Pearson\'s Correlation')
        plt.savefig(os.path.join(save_path, 'r_bootstrap_test.eps'))
        plt.close()


elif args.model == 'classification':
    # Load the dat from the saved pickle
    save_path = '/code/BayOptPy/tpot_%s/Output/vanilla_combi/age/%03d_generations/' \
                    %(args.model, args.generations)

    with open(os.path.join(save_path, 'tpot_all_seeds.pckl'), 'rb') as handle:
        tpot_results = pickle.load(handle)

    with open(os.path.join(save_path, 'rvc_all_seeds.pckl'), 'rb') as handle:
        rvc_results = pickle.load(handle)

    # Do some statistics to see if the results from tpot is significantly differen from rvr
    print('--------------------------------------------------------')
    print('Confusion Matrix - Test dataset')
    print('--------------------------------------------------------')
    t, prob = ttest_ind(tpot_results['confusion_matrix_test'],
                        rvc_results['confusion_matrix_test'], axis=0)
    print('T-statistics:')
    print(t)
    print('p-value: ')
    print(prob)


    plot_confusion_matrix_boosting(
                    np.mean(tpot_results['confusion_matrix_test'], axis=0),
                    np.std(tpot_results['confusion_matrix_test'], axis=0),
                    classes=classes,
                    title='TPOT_test')

    plt.savefig(os.path.join(save_path, 'tpot_test_boosting.eps'))

    plot_confusion_matrix_boosting(
                    np.mean(rvc_results['confusion_matrix_test'], axis=0),
                    np.std(rvc_results['confusion_matrix_test'], axis=0),
                    classes=classes,
                    title='RVC_test')
    plt.savefig(os.path.join(save_path, 'rvc_test_boosting.eps'))


    print('--------------------------------------------------------')
    print('Accuracy - Test dataset')
    print('--------------------------------------------------------')

    print('Mean Accuracy - tpot:')
    print(tpot_results['score_test'])
    print('Mean Accuracy - rvc:')
    print(rvc_results['score_test'])
    t, prob = ttest_ind(tpot_results['score_test'],
                        rvc_results['score_test'], axis=0)
    print('TPOT - boostrap: %.3f +- %.3f' %(np.mean(tpot_results['score_test']),
                                           np.std(tpot_results['score_test'])))
    print('RVC - boostrap: %.3f +- %.3f' %(np.mean(rvc_results['score_test']),
                                           np.std(rvc_results['score_test'])))
    print('T-statistics:')
    print(t)
    print('p-value: ')
    print(prob)

    print('--------------------------------------------------------')
    print('Accuracy - Validation dataset')
    print('--------------------------------------------------------')
    print('Mean Accuracy - tpot: ')
    print(tpot_results['score_test'])
    print('Mean Accuracy - rvc:')
    print(rvc_results['score_test'])
