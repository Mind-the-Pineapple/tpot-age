import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

from BayOptPy.helperfunctions import (set_publication_style,
                                      plot_confusion_matrix_boosting)


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
ind = np.arange(2)
set_publication_style()
# analysis = 'regression'
analysis = 'classification'
classes = np.array(['young', 'old', 'adult'], dtype='U10')

if analysis == 'regression':
    # Load the dat from the saved pickle
    save_path = '/code/BayOptPy/tpot_%s/Output/vanilla_combi/100_generations' %analysis
    with open(os.path.join(save_path, 'tpot_all_seeds.pckl'), 'rb') as handle:
        tpot_results = pickle.load(handle)

    with open(os.path.join(save_path, 'rvr_all_seeds.pckl'), 'rb') as handle:
        rvr_results = pickle.load(handle)

    # Validation plot
    #----------------------------------------------------------------------------
    # Do some statistics to see if the results from tpot is significantly differen from rvr
    print('Test dataset')
    t, prob = ttest_ind(tpot_results['mae_test'], rvr_results['mae_test'])
    print('T-statistics: %.3f, p-value: %.10f' %(t, prob))

    plt.figure()
    plt.bar(ind,
            [np.mean(tpot_results['mae_test']), np.mean(rvr_results['mae_test'])],
            yerr=[np.std(tpot_results['mae_test']),
                  np.std(tpot_results['mae_test'])],
            color=['b', 'r']
                 )
    barplot_annotate_brackets(0, 1, 'p<.001', ind,
                              height=[np.mean(tpot_results['mae_test']),
                                       np.mean(rvr_results['mae_test'])])
    plt.xticks(ind, ('TPOT', 'RVR'))
    plt.ylim([4.5, 7])
    plt.yticks(np.arange(4.5, 7.25, .25))
    plt.ylabel('MAE')
    plt.savefig(os.path.join(save_path, 'MAE_bootstrap_test.png'))
    plt.close()

    # Validation plot
    #----------------------------------------------------------------------------
    print('Validation dataset')
    t, prob = ttest_ind(tpot_results['mae_validation'], rvr_results['mae_validation'])
    print('T-statistics: %.3f, p-value: %.25f' %(t, prob))

    plt.figure()
    plt.bar(ind,
            [np.mean(tpot_results['mae_validation']),
             np.mean(rvr_results['mae_validation'])],
            yerr=[np.std(tpot_results['mae_validation']),
                  np.std(tpot_results['mae_validation'])],
            color=['b', 'r']
                 )
    plt.xticks(ind, ('TPOT', 'RVR'))
    plt.ylim([4.5, 7])
    plt.yticks(np.arange(4.5, 7.25, .25))
    barplot_annotate_brackets(0, 1, 'p<.001', ind,
                              height=[np.mean(tpot_results['mae_validation']),
                                       np.mean(rvr_results['mae_validation'])])
    plt.ylabel('MAE')
    plt.savefig(os.path.join(save_path, 'MAE_bootstrap_validation.png'))

elif analysis == 'classification':
    # Load the dat from the saved pickle
    save_path = '/code/BayOptPy/tpot_%s/Output/vanilla_combi/100_generations/' %analysis

    with open(os.path.join(save_path, 'tpot_all_seeds.pckl'), 'rb') as handle:
        tpot_results = pickle.load(handle)

    with open(os.path.join(save_path, 'rvc_all_seeds.pckl'), 'rb') as handle:
        rvc_results = pickle.load(handle)

    # Do some statistics to see if the results from tpot is significantly differen from rvr
    print('--------------------------------------------------------')
    print('Test dataset')
    print('--------------------------------------------------------')
    t, prob = ttest_ind(tpot_results['confusion_matrix_test'],
                        rvc_results['confusion_matrix_test'], axis=0)
    print('T-statistics:')
    print(t)
    print('p-value: ')
    print(prob)

    print('--------------------------------------------------------')
    print('Validation dataset')
    print('--------------------------------------------------------')
    t, prob = ttest_ind(tpot_results['confusion_matrix_validation'],
                        rvc_results['confusion_matrix_validation'], axis=0)
    print('T-statistics:')
    print(t)
    print('p-value: ')
    print(prob)


    plot_confusion_matrix_boosting(
                    np.mean(tpot_results['confusion_matrix_test'], axis=0),
                    np.std(tpot_results['confusion_matrix_test'], axis=0),
                    classes=classes,
                    title='TPOT_test')

    plt.savefig(os.path.join(save_path, 'tpot_test_boosting.pdf'))

    plot_confusion_matrix_boosting(
                    np.mean(rvc_results['confusion_matrix_test'], axis=0),
                    np.std(rvc_results['confusion_matrix_test'], axis=0),
                    classes=classes,
                    title='RVC_test')
    plt.savefig(os.path.join(save_path, 'rvc_test_boosting.pdf'))

    plot_confusion_matrix_boosting(
                    np.mean(tpot_results['confusion_matrix_validation'], axis=0),
                    np.std(tpot_results['confusion_matrix_validation'], axis=0),
                    classes=classes,
                    title='TPOT_validation')

    plt.savefig(os.path.join(save_path, 'tpot_validation_boosting.pdf'))

    plot_confusion_matrix_boosting(
                    np.mean(rvc_results['confusion_matrix_validation'], axis=0),
                    np.std(rvc_results['confusion_matrix_validation'], axis=0),
                    classes=classes,
                    title='RVC_validation')
    plt.savefig(os.path.join(save_path, 'rvc_validation_boosting.pdf'))
