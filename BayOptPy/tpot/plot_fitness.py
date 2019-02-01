import os
import pickle
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns;
sns.set()
import argparse

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
tpot_path = os.path.join(project_wd, 'BayOptPy', 'tpot')

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

# plot the fitness over different genrations
fig, ax = plt.subplots(1)
plt_filled_std(ax, range(len(fitness)), fitness['avg'], fitness['std'], colour_list[0])
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.savefig(os.path.join(tpot_path, 'fitness.png'))
