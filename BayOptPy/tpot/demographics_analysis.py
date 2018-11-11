import os
import pandas as pd
import pdb
import argparse

from BayOptPy.helperfunctions import get_paths

parser = argparse.ArgumentParser()
parser.add_argument('-dataset',
                    dest='dataset',
                    help='Specify which dataset to use',
                    choices=['OASIS', 'BANC']
                    )
args = parser.parse_args()

if args.dataset == 'BANC':
    # BANC analysis
    print('Analysing the BANC dataset')
    debug = True
    dataset = 'BANC'
    project_wd, project_data, project_sink = get_paths(debug, dataset)

    demographics_csv = os.path.join(project_data, 'BANC_2016.csv')
    df = pd.read_csv(demographics_csv, header=None, names=['ID', 'Dataset', 'Sex', 'Age'])

    # Calculate the dataset mean_age
    mean_age = df['Age'].mean()
    std_age = df['Age'].std()
    print('Mean Age %.2f +- %.2f' %(mean_age, std_age))
else:
    print('Analysing the OASIS dataset')
    debug = True
    dataset = 'OASIS'
    project_wd, project_data, project_sink = get_paths(debug, dataset)
    demographics_csv = os.path.join(project_data, 'oasis_cross-sectional.csv')
    df = pd.read_csv(demographics_csv)

    # Calculate the dataset mean_age
    mean_age = df['Age'].mean()
    std_age = df['Age'].std()
    print('Mean Age %.2f +- %.2f' %(mean_age, std_age))


pdb.set_trace()
