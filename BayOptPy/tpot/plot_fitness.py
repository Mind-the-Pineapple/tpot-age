import os
import pickle
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
sns.set()
import argparse
import numpy as np
import pandas as pd
import joblib

from BayOptPy.helperfunctions import (get_paths, get_all_random_seed_paths,
                                      get_mae_for_all_generations,
                                      set_publication_style)

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
                             'feat_selec', 'vanilla', 'vanilla_combi',
                             'mutation', 'random_seed'],
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
parser.add_argument('-mutation_rate',
                   dest='mutation_rate',
                   help='Must be on the range [0, 1.0]',
                   type=float,
                   default=.9
                   )
parser.add_argument('-crossover_rate',
                    dest='crossover_rate',
                    help='Cross over of the genetic algorithm. Must be on \
                    the range [0, 1.0]',
                    type=float,
                    default=.1
                   )
args = parser.parse_args()

# Set plot styles
set_publication_style()

random_seeds = [20, 30, 40, 50, 60]

# get correct path
project_wd, project_data, project_sink = get_paths(args.debug, args.dataset)
tpot_path = get_all_random_seed_paths(args.analysis, args.generations,
                                      args.population_size,
                                      args.debug,
                                      args.mutation_rate,
                                      args.crossover_rate)

colour_list = ['#5dade2', '#e67e22']
def plt_filled_std(ax, data, data_mean, data_std, color, label=None):
    cis = (data_mean - data_std, data_mean + data_std)
    # plot filled area
    ax.fill_between(data, cis[0], cis[1], alpha=.2, color=color)
    # plot mean
    ax.plot(data, data_mean, linewidth=2, label=label)
    ax.margins(x=0)

# load file
mae_train_all = []
mae_test_all = []
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
    plt_filled_std(ax, range(len(fitness)), fitness['avg'], fitness['std'],
                   colour_list[0], None)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.savefig(os.path.join(generation_analysis_path, 'mean_std.png'))
    plt.close()


    # plot the max fitness over different generations for the training and test dataset
    all_mae_test_set, all_mae_train_set, pipeline_complexity = \
            get_mae_for_all_generations(args.dataset,
                                        random_seed,
                                        args.generations,
                                        args.config_dict, tpot_path)
    mae_train_all.append(all_mae_train_set)
    mae_test_all.append(all_mae_test_set)
    plt.figure()
    plt.plot(range(len(all_mae_train_set)), all_mae_train_set, marker='o',
             color=colour_list[0], label='traning set')
    # plt.plot(range(len(fitness)), fitness['avg'], marker='o',
    #          label='avg_training')
    plt.plot(range(len(all_mae_test_set)), all_mae_test_set, marker='o',
             color=colour_list[1], label='test set')
    plt.legend()
    plt.xlabel('Generation')
    plt.ylabel('MAE')
    plt.ylim(5,8)
    plt.savefig(os.path.join(generation_analysis_path, 'train_test_fitness.png'))

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



    # plot the cross-validated MAE for the training and test dataset
    plt.figure()
    plt.plot(range(len(fitness)), abs(fitness['max']), marker='o', label='traning set')
    # plt.plot(range(len(fitness)), fitness['avg'], marker='o',
    #          label='avg_training')
    plt.legend()
    plt.xlabel('Generation')
    plt.ylabel('MAE')
    plt.savefig(os.path.join(generation_analysis_path, 'max_fitness.png'))
    # Save the current random_see max fitness for further analysis

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
    data = [abs(fitness['raw'][generation]) for generation in
                         range(len(fitness))]
    fig, ax = plt.subplots(1,1)
    outliers = dict(markerfacecolor='#FFA500', marker='o', alpha=.1)
    plt.boxplot(data, positions=range(0, len(fitness)), showfliers=True, flierprops=outliers)
    plt.ylabel('MAE')
    plt.xlabel('Generations')
    # TODO: improve how you determine this threshold (there are models that are worse)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.savefig(os.path.join(generation_analysis_path, 'boxplot.png'))
    plt.close()

    #Plot Boxplot
    plt.figure()
    fig, ax = plt.subplots(1,1)
    outliers = dict(markerfacecolor='#FFA500', marker='o', alpha=.1)
    plt.boxplot(data, positions=range(0, len(fitness)), showfliers=True, flierprops=outliers)
    plt.ylabel('MAE')
    plt.xlabel('Generations')
    # TODO: improve how you determine this threshold (there are models that are worse)
    plt.ylim(0, 45)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.savefig(os.path.join(generation_analysis_path, 'boxplot2.png'))
    plt.close()

    #Plot Boxplot at every 10-th generation
    plt.figure()
    fig, ax = plt.subplots(1,1)
    outliers = dict(markerfacecolor='#FFA500', marker='o', alpha=.1)

    plt.boxplot(data[0:101:10], positions=range(0, len(fitness),10), showfliers=True, flierprops=outliers)
    plt.ylabel('MAE')
    plt.xlabel('Generations')
    # TODO: improve how you determine this threshold (there are models that are worse)
    plt.ylim(0, 45)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.savefig(os.path.join(generation_analysis_path, 'boxplot3.png'))
    plt.close()

    # Plot violin plot
    plt.figure()
    fig, ax = plt.subplots(1, 1)
    plt.violinplot(data, positions=range(0, len(fitness)), showmedians=True)
    plt.ylabel('MAE')
    plt.xlabel('Generations')
    plt.savefig(os.path.join(generation_analysis_path, 'violin.png'))
    plt.close()

    # Plot violin plot2
    plt.figure()
    fig, ax = plt.subplots(1, 1)
    plt.violinplot(data, positions=range(0, len(fitness)), showmedians=True)
    plt.ylabel('MAE')
    plt.xlabel('Generations')
    plt.ylim(4, 45)
    plt.savefig(os.path.join(generation_analysis_path, 'violin2.png'))
    plt.close()

    # plot the MAE for a selected number of generations
    selected_gens = np.arange(0,args.generations+1,5)
    tpot_obj_path = os.path.join(tpot_path, 'random_seed_%03d',
                                  'tpot_BANC_freesurf_%s_%03dgen.dump'
                                 ) %(random_seed, args.config_dict, args.generations)
    tpot_obj = joblib.load(tpot_obj_path)
    for selected_gen in selected_gens:
        print(selected_gen)
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

        print('First MAE Group: samller -9')
        print('Number of models: %d' %len(cluster_mae_name[0]))
        print(cluster_mae_name[0])
        print('Second MAE Group: x > 19')
        print('Number of models: %d' %len(cluster_mae_name[1]))
        print(cluster_mae_name[1])
        print('Third MAE Group: between 9 and 19')
        print('Number of models: %d' %len(cluster_mae_name[2]))
        print(cluster_mae_name[2])
        print('------------------------------------------------------------------')

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
    print('')
    print('------------------------------------------------------------------')
    print('Plot Heatmap with algorithm counts')
    print('------------------------------------------------------------------')
    print('')
    # Define the list of possible models
    algorithms_list = ['GaussianProcessRegressor', 'RVR', 'LinearSVR',
                                'RandomForestRegressor',
                                'KNeighborsRegressor',
                                'LinearRegression', 'Ridge','ElasticNetCV',
                                'ExtraTreesRegressor', 'LassoLarsCV',
                                'DecisionTreeRegressor']

    df = pd.DataFrame(columns=algorithms_list,
                     index=tpot_obj['evaluated_individuals'].keys())

    # Iterate over all the dictionary keys from the dictionary with the TPOT
    # analysed model and count the ocurrence of each one of the algorithms of
    # interest
    for generation in tpot_obj['evaluated_individuals']:
        algorithms_dic = dict.fromkeys(algorithms_list, 0)
        for algorithm in algorithms_list:
            for tpot_pipeline in tpot_obj['evaluated_individuals'][generation]:
                 # By using '( we count the  algorithm only once and do not# care about its
                 # hyperparameters definitions'
                 algorithms_dic[algorithm] += tpot_pipeline.count(algorithm + '(')
                 # Append the count of algorithms to the dataframe
                 df.loc[generation] = pd.Series(algorithms_dic)

     # Create heatmap
    df2 = df.transpose()
    plt.figure(figsize=(8,8))
    sns.heatmap(df2,
             cmap='YlGnBu')
    plt.xlabel('Generations')
    plt.tight_layout()
    plt.savefig(os.path.join(generation_analysis_path, 'heatmap.png'))
    plt.close()

    print('Maximum Count for each algorithm')
    print(df2.max())
    print('Minimum Count for each algorithm')
    print(df2.min())


# Plot max statiscal max fitness for the different random seeds
# plot the mean and std of the fitness over different generations
mae_test_all_np = np.array(mae_test_all)
mae_train_all_np = np.array(mae_train_all)
fig, ax = plt.subplots(1)
plt_filled_std(ax, range(mae_test_all_np.shape[1]), np.mean(mae_test_all_np, axis=0),
               np.std(mae_test_all_np, axis=0), colour_list[1], 'Test')
# plt_filled_std(ax, range(mae_train_all_np.shape[1]), np.mean(mae_train_all_np, axis=0),
#                np.std(mae_train_all_np, axis=0), colour_list[0], 'Train')
plt.ylabel('Generation')
plt.xlabel('MAE')
plt.legend()
plt.savefig(os.path.join(tpot_path, 'all_seeds_mean_std.png'))
plt.close()

#Plot Boxplot
plt.figure()
fig, ax = plt.subplots(1,1)
outliers = dict(markerfacecolor='#FFA500', marker='o', alpha=.1)
plt.boxplot(mae_train_all, positions=range(0, len(mae_train_all)),
            showfliers=True, flierprops=outliers)
plt.ylabel('MAE')
plt.xlabel('Random Seeds')
# ax.set_yticks(args.random_seeds)
plt.savefig(os.path.join(tpot_path, 'boxplot_all_random_train.png'))
plt.close()


