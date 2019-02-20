import os
import pickle
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns;
sns.set()
import argparse
import numpy as np
import math
import re

from BayOptPy.helperfunctions import get_paths, get_all_random_seed_paths

def set_publication_style():
    # Se font size to paper size
    sns.set_context('paper')

    # Set the font to be serif, rahter than sans
    sns.set(font='serif')

    # Make background white, and specify the specific font family
    sns.set_style("white", {
                      "font.family": "serif",
                      "font.serif": ["Times", "Palatino", "serif"]
                                    })

def get_mae_for_all_generations():
    '''
    Get the MAE values for both the training and test dataset
    :return:
    '''
    # Load the scores for the best models
    checkpoint_path = os.path.join(tpot_path, 'random_seed_%03d' %random_seed, 'checkpoint_folder')
    # Find the saved dictionary with the MAE for each generation and load the MAE on the test-dataset
    # Note that if a value is not present for a generation, that means that the score did not change from the previous
    # generation
    fileList = os.listdir(checkpoint_path)
    saved_files = [re.sub(r'^pipeline_log_gen_(.*?)\_.pckl', '\\1', file) for
                   file in fileList if file.endswith('.pckl')]
    # sort the array in ascending order
    saved_files.sort()
    mae_test = []
    mae_train = []
    gen = []
    for file in saved_files:
        with open(os.path.join(checkpoint_path, file), 'rb') as handle:
            tmp = pickle.load(handle)
        mae_test.append(tmp['pipeline_test_mae'])
        mae_train.append(tmp['pipeline_score'])
        gen.append(tmp['gen'])

    # Iterate over the the list of saved MAEs and repeat the values where one generation is missed
    all_mae_test = []
    all_mae_train = []
    curr_gen_idx = 0
    # all generations
    for generation in range(args.generations):
        if generation == gen[curr_gen_idx]:
            all_mae_test.append(mae_test[curr_gen_idx])
            all_mae_train.append(mae_train[curr_gen_idx])
            if len(mae_test) > 1 and (len(gen) > curr_gen_idx + 1):
                curr_gen_idx += 1
        else:
            # repeat the same last value
            all_mae_test.append(all_mae_test[-1])
            all_mae_train.append(all_mae_train[-1])
    return all_mae_test, all_mae_train

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
                    choices=['vanilla', 'population'],
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
                   )
args = parser.parse_args()

# Set plot styles
#set_publication_style()

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
    all_mae_test_set, all_mae_train_set = get_mae_for_all_generations()
    plt.figure()
    plt.plot(range(len(all_mae_train_set)), all_mae_train_set, marker='o', label='traning set')
    # plt.plot(range(len(fitness)), fitness['avg'], marker='o',
    #          label='avg_training')
    plt.plot(range(len(all_mae_test_set)), all_mae_test_set, marker='o', label='test set')
    plt.legend()
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Random Seed %d' %random_seed)
    plt.savefig(os.path.join(generation_analysis_path, 'train_test_fitness.png'))
    # Save the current random_see max fitness for further analysis
    avg_max_fitness.append(fitness['max'])

    # plot the cross-validated MAE for the training and test dataset
    all_mae_test_set = get_mae_for_all_generations()
    plt.figure()
    plt.plot(range(len(fitness)), fitness['max'], marker='o', label='traning set')
    # plt.plot(range(len(fitness)), fitness['avg'], marker='o',
    #          label='avg_training')
    plt.legend()
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
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
    for generation in range(len(fitness)):
        plt.figure()
        plt.hist(fitness['raw'][generation], bins=50, range=(np.min(fitness['min']), np.max(fitness['max'])), histtype='bar')
        #yint = range(int(min_n), int(max_n+2), 10)
        yint = range(0, -60, 10)
        plt.yticks(yint)
        plt.xlabel('Fitness')
        plt.ylabel('Counts')
        plt.title('Generation %d' %generation)
        plt.savefig(os.path.join(generation_analysis_path, 'histo_%d.png') %generation)
        plt.close()


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
    data = [fitness['raw'][generation] for generation in range(len(fitness))]
    plt.title('Basic Plot')
    fig, ax = plt.subplots(1,1)
    plt.boxplot(data, positions=range(0, len(fitness)))
    plt.ylabel('Fitness')
    plt.xlabel('Generations')
    plt.title('Random Seed %d' %random_seed)
    # TODO: improve how you determine this threshold (there are models that are worse)
    plt.ylim(-45, 0)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.savefig(os.path.join(generation_analysis_path, 'boxplot.png'))
    plt.close()

# Plot max statiscal max fitness for the different random seeds
# plot the mean and std of the fitness over different generations
fig, ax = plt.subplots(1)
plt_filled_std(ax, range(len(fitness)), np.mean(avg_max_fitness, axis=0), np.std(avg_max_fitness, axis=0), colour_list[0])
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.savefig(os.path.join(tpot_path, 'all_seeds_mean_std.png'))
plt.close()



