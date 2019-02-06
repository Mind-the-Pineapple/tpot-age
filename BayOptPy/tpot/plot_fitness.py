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

from BayOptPy.helperfunctions import get_paths

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
args = parser.parse_args()

# get corerct path
project_wd, project_data, project_sink = get_paths(args.debug, args.dataset)
tpot_path = os.path.join(project_wd, 'BayOptPy', 'tpot',)
generation_analysis_path = os.path.join(tpot_path, 'generation_analysis')

# check if saving paths exists otherwise create it
if not os.path.exists(os.path.join(generation_analysis_path)):
    os.makedirs(os.path.join(generation_analysis_path))


colour_list = ['#588ef3']
def plt_filled_std(ax, data, data_mean, data_std, color):
    cis = (data_mean - data_std, data_mean + data_std)
    # plot filled area
    ax.fill_between(data, cis[0], cis[1], alpha=.2, color=color)
    # plot mean
    ax.plot(data, data_mean, linewidth=2)
    ax.margins(x=0)

# load file
with open(os.path.join(tpot_path, 'logbook.pkl'), 'rb') as handle:
    fitness = pickle.load(handle)

# plot the mean and std of the fitness over different generations
fig, ax = plt.subplots(1)
plt_filled_std(ax, range(len(fitness)), fitness['avg'], fitness['std'], colour_list[0])
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.savefig(os.path.join(generation_analysis_path, 'mean_std.png'))

# plot the max fitness over different generations
plt.figure()
plt.plot(range(len(fitness)), fitness['max'], marker='o')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.savefig(os.path.join(generation_analysis_path, 'max_fitness.png'))


# find the maximum and minimum histogram count
max_n = 0
min_n = 0
for generation in range(len(fitness)):
    n, _, _ = plt.hist(fitness['raw'][generation], bins=50)
    if np.max(n) > max_n:
        max_n = np.max(n)
    if np.min(n) < min_n:
        min_n = np.min(n)


# plot Histogram
for generation in range(len(fitness)):
    plt.figure()
    plt.hist(fitness['raw'][generation], bins=50, range=(np.min(fitness['min']), np.max(fitness['max'])), histtype='bar')
    yint = range(int(min_n), int(max_n+2), 10)
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
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

plt.savefig(os.path.join(generation_analysis_path, 'boxplot.png'))
