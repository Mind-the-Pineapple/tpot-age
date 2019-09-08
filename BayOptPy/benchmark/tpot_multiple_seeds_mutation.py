import pickle
from itertools import combinations
import argparse

from scipy.stats import ttest_ind, f_oneway

parser = argparse.ArgumentParser()
generations = 10
analysis = 'preprocessing'
# analysis = 'mutation'
parser.add_argument('-analysis',
                    dest='analysis',
                    help='Specify which type of analysis to use',
                    choices=['mutation',
                             'preprocessing'],
                    required=True
                    )

args = parser.parse_args()

if __name__ == '__main__':
    results = {}
    print(analysis)
    if analysis == 'mutation':
        # Load the summary of accuracy for the different random seeds
        experiments = ['0.1_mut_0.9_cross', '0.5_mut_0.5_cross', '0.9_mut_0.1_cross' ]
        for experiment in experiments:
            save_path = '/code/BayOptPy/tpot_regression/Output/mutation/age/%03d_generations/%s/tpot_all_seeds.pckl' %(generations, experiment)
            # load results
            with open(save_path, 'rb') as handle:
                results[experiment] = pickle.load(handle)

        print('One-way ANOVA: Test')
        print(f_oneway(results['0.1_mut_0.9_cross']['mae_test'],
                       results['0.5_mut_0.5_cross']['mae_test'],
                       results['0.9_mut_0.1_cross']['mae_test']))
        print('One-way ANOVA: Validation')
        print(f_oneway(results['0.1_mut_0.9_cross']['mae_validation'],
                       results['0.5_mut_0.5_cross']['mae_validation'],
                       results['0.9_mut_0.1_cross']['mae_validation']))

        for combination in combinations(experiments, 2):
            print(combination)
            print('T-Test: Test')
            print(ttest_ind(results[combination[0]]['mae_test'],
                            results[combination[1]]['mae_test']))
            print('T-test: Validation')
            print(ttest_ind(results[combination[0]]['mae_validation'],
                            results[combination[1]]['mae_validation']))

    if analysis == 'preprocessing':
       # Load the results for feature combination and feature selection
        experiments = ['feat_selec', 'feat_combi', 'vanilla', 'vanilla_combi']

        for experiment in experiments:
            save_path = '/code/BayOptPy/tpot_regression/Output/%s/age/%03d_generations/tpot_all_seeds.pckl' \
            %(experiment, generations)
            # load results
            with open(save_path, 'rb') as handle:
                results[experiment] = pickle.load(handle)
        print('One-way ANOVA: Test')
        print(f_oneway(results['feat_selec']['mae_test'],
                       results['feat_combi']['mae_test'],
                       results['vanilla_combi']['mae_test'],
                       results['vanilla']['mae_test']))
        print('One-way ANOVA: Validation')
        print(f_oneway(results['feat_selec']['mae_validation'],
                       results['feat_combi']['mae_validation'],
                       results['vanilla_combi']['mae_validation'],
                       results['vanilla']['mae_validation']))


