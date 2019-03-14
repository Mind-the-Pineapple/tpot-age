import os
import pickle
import joblib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
sns.set()
import argparse
import numpy as np
import joblib

from BayOptPy.helperfunctions import get_paths, get_all_random_seed_paths

def set_publication_style():
    # Se font size to paper size
    plt.style.use(['seaborn-white', 'seaborn-talk'])
    matplotlib.rc("font", family="Times New Roman")
    # Remove the spines
    sns.set_style('white', {"axes.spines.top": False,
                            "axes.spines.right": False,
                            "axes.labelsize": 'large'})


def get_mae_for_all_generations(dataset, random_seed, generations, config_dict):
    '''
    Get the MAE values for both the training and test dataset
    :return:
    '''
    # Load the scores for the best models
    saved_path = os.path.join(tpot_path, 'random_seed_%03d' %random_seed,
                                   'tpot_%s_%s_%03dgen_pipelines.dump'
                                    %(dataset, config_dict, generations))
    # Note that if a value is not present for a generation, that means that the
    # score did not change from the previous generation
    # sort the array in ascending order
    logbook = joblib.load(saved_path)
    gen = list(logbook['log'].keys())

    print('There are %d optminal pipelines' %len(gen))
    print('These are the best pipelines')
    for generation in gen:
        print(logbook['log'][generation]['pipeline_name'])

    # Iterate over the the list of saved MAEs and repeat the values where one
    # generation is missed
    all_mae_test = []
    all_mae_train = []
    pipeline_complexity = []
    curr_gen_idx = 0
    # all generations
    for generation in range(generations):
        if generation == gen[curr_gen_idx]:
            all_mae_test.append(abs(logbook['log'][gen[curr_gen_idx]]['pipeline_test_mae']))
            all_mae_train.append(abs(logbook['log'][gen[curr_gen_idx]]['pipeline_score']))
            pipeline_complexity.append(len(logbook['log'][gen[curr_gen_idx]]['pipeline_tree']))
            if len(gen) > 1 and (len(gen) > curr_gen_idx + 1):
                curr_gen_idx += 1
        else:
            # repeat the same last value
            all_mae_test.append(all_mae_test[-1])
            all_mae_train.append(all_mae_train[-1])
            pipeline_complexity.append(pipeline_complexity[-1])

    # transform the pipeline_complexity into a numpy array, in order to perform
    # fancy indexing
    pipeline_complexity = np.array(pipeline_complexity)
    return all_mae_test, all_mae_train, pipeline_complexity

parser = argparse.ArgumentParser()
parser.add_argument('-debug',
                   dest='debug',
                   action='store_true',
                   help='Run debug with Pycharm'
                   )
parser.add_argument('-dataset',
                   dest='dataset',
                   help='Specify which dataset to use',
                   choices=['OASIS', 'BANC',
                            'BANC_freesurf']
                   )
parser.add_argument('-analysis',
                    dest='analysis',
                    help='Specify which type of analysis to use',
                    choices=['vanilla', 'population', 'feat_combi',
                             'feat_selec', 'vanilla', 'vanilla_combi'],
                    required=True
                    )
parser.add_argument('-generations',
                    dest='generations',
                    help='Specify number of generations to use',
                    type=int,
                    required=True
                    )
parser.add_argument('-population_size',
                    dest='population_size',
                    help='Specify population size to use',
                    type=int,
                    default=100 # use the same default as TPOT default population value
                    )
parser.add_argument('-random_seeds',
                    dest='random_seeds',
                    help='Specify list of random seeds to use',
                    nargs='+',
                    required=True,
                    type=int
                   ),
parser.add_argument('-config_dict',
                    dest='config_dict',
                    help='Specify the list of models to use',
                    required=True,
                    choices=['None', 'light', 'custom', 'gpr', 'gpr_full',
                        'feat_combi', 'feat_selec', 'vanilla', 'vanilla_combi']
                   )
args = parser.parse_args()

# Set plot styles
set_publication_style()

#random_seeds = [0, 5, 10, 20, 30, 42, 60, 80]
#random_seeds = [30]

# get corerct path
project_wd, project_data, project_sink = get_paths(args.debug, args.dataset)
tpot_path = get_all_random_seed_paths(args.analysis, args.generations,
                                      args.population_size,
                                      args.debug)

colour_list = ['#588ef3']
def plt_filled_std(ax, data, data_mean, data_std, color):
    cis = (data_mean - data_std, data_mean + data_std)
    # plot filled area
    ax.fill_between(data, cis[0], cis[1], alpha=.2, color=color)
    # plot mean
    ax.plot(data, data_mean, linewidth=2)
    ax.margins(x=0)

# load file
avg_max_fitness = []
for random_seed in args.random_seeds:
    generation_analysis_path = os.path.join(tpot_path, 'random_seed_%03d', 'generation_analysis') %random_seed

    # check if saving paths exists otherwise create it
    if not os.path.exists(os.path.join(generation_analysis_path)):
        os.makedirs(os.path.join(generation_analysis_path))

    # Load dumped file
    with open(os.path.join(tpot_path, 'logbook_rnd_seed%03d.pkl') %random_seed, 'rb') as handle:
        fitness = pickle.load(handle)

    # plot the mean and std of the fitness over different generations
    fig, ax = plt.subplots(1)
    plt_filled_std(ax, range(len(fitness)), fitness['avg'], fitness['std'], colour_list[0])
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.savefig(os.path.join(generation_analysis_path, 'mean_std.png'))
    plt.close()


    # plot the max fitness over different generations for the training and test dataset
    all_mae_test_set, all_mae_train_set, pipeline_complexity = \
            get_mae_for_all_generations(args.dataset,
                                        random_seed,
                                        args.generations,
                                        args.config_dict)
    plt.figure()
    plt.plot(range(len(all_mae_train_set)), all_mae_train_set, marker='o', label='traning set')
    # plt.plot(range(len(fitness)), fitness['avg'], marker='o',
    #          label='avg_training')
    plt.plot(range(len(all_mae_test_set)), all_mae_test_set, marker='o', label='test set')
    plt.legend()
    plt.xlabel('Generation')
    plt.ylabel('MAE')
    plt.title('Random Seed %d' %random_seed)
    plt.savefig(os.path.join(generation_analysis_path, 'train_test_fitness.png'))
    # Save the current random_see max fitness for further analysis
    avg_max_fitness.append(fitness['max'])

    # Plot the different MAE for each generation and use the complexity of the
    # model as hue
    # plt.figure()
    # import pdb
    # pdb.set_trace()
    # n_complexities = np.unique(pipeline_complexity)
    # for complexity in n_complexities:
    #     plt.scatter(range(len(pipeline_complexity)),
    #             all_mae_train_set[n_complexities==complexity], marker='o',
    #             c=complexity, label=complexity)
    # plt.xlabel('Generation')
    # plt.ylabel('MAE')
    # plt.legend()
    # plt.savefig(os.path.join(generation_analysis_path, 'pipeline_complexity.png'))

    # Plot complexity as a bar plot for each generation
    plt.figure()
    plt.plot(range(len(pipeline_complexity)), pipeline_complexity)
    plt.xlabel('Generation')
    plt.ylabel('Pipeline complexity')
    plt.savefig(os.path.join(generation_analysis_path, 'pipeline_complexity.png'))


    # plot the cross-validated MAE for the training and test dataset
    plt.figure()
    plt.plot(range(len(fitness)), abs(fitness['max']), marker='o', label='traning set')
    # plt.plot(range(len(fitness)), fitness['avg'], marker='o',
    #          label='avg_training')
    plt.legend()
    plt.xlabel('Generation')
    plt.ylabel('MAE')
    plt.title('Random Seed %d' %random_seed)
    plt.savefig(os.path.join(generation_analysis_path, 'max_fitness.png'))
    # Save the current random_see max fitness for further analysis
    avg_max_fitness.append(fitness['max'])

    # # find the maximum and minimum histogram count
    # max_n = 0
    # min_n = 0
    # for generation in range(len(fitness)):
    #     n, _, _ = plt.hist(fitness['raw'][generation], bins=50)
    #     if np.max(n) > max_n:
    #         max_n = np.max(n)
    #     if np.min(n) < min_n:
    #         min_n = np.min(n)


    # plot Histogram
    # for generation in range(len(fitness)):
    #     plt.figure()
    #     plt.hist(fitness['raw'][generation], bins=50, range=(np.min(fitness['min']), np.max(fitness['max'])), histtype='bar')
    #     #yint = range(int(min_n), int(max_n+2), 10)
    #     yint = range(0, -60, 10)
    #     plt.yticks(yint)
    #     plt.xlabel('Fitness')
    #     plt.ylabel('Counts')
    #     plt.title('Generation %d' %generation)
    #     plt.savefig(os.path.join(generation_analysis_path, 'histo_%d.png') %generation)
    #     plt.close()


    # Plot All histograms in one plot
    # plt.figure()
    # max_n = 0
    # min_n = 0
    # for generation in range(len(fitness)):
    #     n, bins, _ = plt.hist(fitness['raw'][generation], bins=50, range=(np.min(fitness['min']), np.max(fitness['max'])),
    #                           histtype='step', label='Generation %d' %generation)
    #     if np.max(n) > max_n:
    #         max_n = np.max(n)
    #     if np.min(n) < min_n:
    #         min_n = np.min(n)
    # yint = range(int(min_n), math.ceil(int(max_n + 1)))
    # plt.yticks(yint)
    # plt.legend()
    # plt.xlabel('Fitness')
    # plt.ylabel('Counts')
    # plt.title('Generation %d' %generation)
    # plt.savefig(os.path.join(tpot_path, 'histo_all.png'))
    # plt.close()

    #Plot Boxplot
    plt.figure()
    data = [abs(fitness['raw'][generation]) for generation in range(len(fitness))]
    plt.title('Basic Plot')
    fig, ax = plt.subplots(1,1)
    outliers = dict(markerfacecolor='#FFA500', marker='o', alpha=.5)
    plt.boxplot(data, positions=range(0, len(fitness)), vert=False, showfliers=True, flierprops=outliers)
    plt.xlabel('MAE')
    plt.ylabel('Generations')
    plt.title('Random Seed %d' %random_seed)
    # TODO: improve how you determine this threshold (there are models that are worse)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.savefig(os.path.join(generation_analysis_path, 'boxplot.png'))
    plt.close()

    #Plot Boxplot
    plt.figure()
    plt.title('Basic Plot')
    fig, ax = plt.subplots(1,1)
    outliers = dict(markerfacecolor='#FFA500', marker='o', alpha=.5)
    plt.boxplot(data, positions=range(0, len(fitness)), vert=False, showfliers=True, flierprops=outliers)
    plt.xlabel('MAE')
    plt.ylabel('Generations')
    plt.title('Random Seed %d' %random_seed)
    # TODO: improve how you determine this threshold (there are models that are worse)
    plt.xlim(4, 45)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.savefig(os.path.join(generation_analysis_path, 'boxplot2.png'))
    plt.close()

    # Plot violin plot
    plt.figure()
    fig, ax = plt.subplots(1, 1)
    plt.violinplot(data, positions=range(0, len(fitness)), vert=False, showmedians=True)
    plt.xlabel('MAE')
    plt.ylabel('Generations')
    plt.title('Random Seed %d' % random_seed)
    plt.savefig(os.path.join(generation_analysis_path, 'violin.png'))
    plt.close()

    # Plot violin plot2
    plt.figure()
    fig, ax = plt.subplots(1, 1)
    plt.violinplot(data, positions=range(0, len(fitness)),vert=False, showmedians=True)
    plt.xlabel('MAE')
    plt.ylabel('Generations')
    plt.title('Random Seed %d' % random_seed)
    plt.xlim(4, 45)
    plt.savefig(os.path.join(generation_analysis_path, 'violin2.png'))
    plt.close()

    # plot the MAE for a selected number of generations
    selected_gens = np.arange(0,args.generations+1,5)
    tpot_obj_path = os.path.join(tpot_path, 'random_seed_%03d',
                                  'tpot_BANC_freesurf_%s_%03dgen.dump'
                                 ) %(random_seed, args.config_dict, args.generations)
    tpot_obj = joblib.load(tpot_obj_path)
    for selected_gen in selected_gens:
        curr_gen_mae = \
        [tpot_obj['evaluated_individuals'][selected_gen][model]['internal_cv_score']
                for model in
                         tpot_obj['evaluated_individuals'][selected_gen].keys()]
        # Devide mae into 3 groups according to their values
        group_selected_gen_mae = [0 if x>-9 else 1 if x<-19 else 2 for x in
                               curr_gen_mae]
        print('Show group belongings')
        print(group_selected_gen_mae)


        # Print the models in the first groups
        cluster_mae_name = {k:[] for k in range(3)}
        cluster_mae = {k:[] for k in range(3)}
        cluster_idx = {k:[] for k in range(3)}
        # Note: From Python 3.6 forwards dictionary are ordered
        for idx, key in enumerate(tpot_obj['evaluated_individuals'][selected_gen]):
            if group_selected_gen_mae[idx] == 0:
                cluster_idx[0].append(idx)
                cluster_mae_name[0].append(key)
                cluster_mae[0].append(abs(curr_gen_mae[idx]))

            if group_selected_gen_mae[idx] == 1:
                cluster_idx[1].append(idx)
                cluster_mae_name[1].append(key)
                cluster_mae[1].append(abs(curr_gen_mae[idx]))

            if group_selected_gen_mae[idx] == 2:
                cluster_idx[2].append(idx)
                cluster_mae_name[2].append(key)
                cluster_mae[2].append(abs(curr_gen_mae[idx]))

        print('First MAE Group:')
        print('Number of models: %d' %len(cluster_mae_name[0]))
        print(cluster_mae_name[0])
        print('Second MAE Group')
        print('Number of models: %d' %len(cluster_mae_name[1]))
        print(cluster_mae_name[1])
        print('Third MAE Group')
        print('Number of models: %d' %len(cluster_mae_name[2]))
        print(cluster_mae_name[2])

        plt.figure()
        markers = 'o'
        hatchs = [None, '|', '|||']
        sns.set_palette('OrRd')
        for group in np.unique(group_selected_gen_mae):
            plt.scatter(cluster_idx[group], cluster_mae[group], marker=markers, hatch=hatchs[group])
        plt.ylabel('MAE')
        plt.xlabel('Models')
        plt.savefig(os.path.join(generation_analysis_path, '%d_gen_mae.png'
            %selected_gen))
        plt.close()

# Plot max statiscal max fitness for the different random seeds
# plot the mean and std of the fitness over different generations
fig, ax = plt.subplots(1)
plt_filled_std(ax, range(len(fitness)), np.mean(avg_max_fitness, axis=0), np.std(avg_max_fitness, axis=0), colour_list[0])
plt.xlabel('Generation')
plt.ylabel('MAE')
plt.savefig(os.path.join(tpot_path, 'all_seeds_mean_std.png'))
plt.close()



