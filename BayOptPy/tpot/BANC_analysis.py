import argparse
from tpot import import TPOTRegressor
from sklearn import import model_selection
import numpy as np
from dask.distributed import import Client

from BayOptPy.helperfunctions import import get_data, get_paths

parser = argparse.ArgumentParser()
parser.add_argument('-nogui',
                    dest='nogui',
                    action='store_true',
                    help='No gui'
                    )
parser.add_argument('-debug',
                    dest='debug',
                    action='store_true',
                    help='Run debug with Pycharm'
                    )
parser.add_argument('-dask',
                    dest='dask',
                    action='store_true',
                    help='Run analysis with dask'
                    )
