import pickle
import argparse
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from tpot.builtins import ZeroCount
from sklearn.model_selection import cross_val_predict

from BayOptPy.helperfunctions import get_data, get_paths

parser = argparse.ArgumentParser()
parser.add_argument('-debug',
                    dest='debug',
                    action='store_true',
                    help='Run debug with Pycharm'
                    )
parser.add_argument('-dataset',
                    dest='dataset',
                    help='Specify which dataset to use',
                    choices=['OASIS', 'BANC']
                   )
parser.add_argument('-resamplefactor',
                    dest='resamplefactor',
                    help='Specify resampling rate for the image affine',
                    type=int,
                    default=1 # no resampling is performed
                    )
args = parser.parse_args()

if __name__ == '__main__':
    print('The current arguments are: %s' %args)

    # load pickle with the list of analysed models
    with open('BayOptPy/tpot/OASIS_pipelines.pkl', 'rb') as handle:
        analysed_dic = pickle.load(handle)

    # load the brain imaging data
    project_wd, project_data, project_sink = get_paths(args.debug,
                                                       args.dataset)
    demographics, imgs, maskedData = get_data(project_data,
                                              args.dataset, args.debug,
                                              project_wd,
                                              args.resamplefactor)

    # transform analysed_models into a list and iterate over it
    predicted_age = []
    analysed_models = list(analysed_dic.items())
    for model in analysed_models:
        model_name = model[0]
        model_info = model[1]

        # fit the data
        exported_pipeline = make_pipeline(
            ZeroCount(),
            MinMaxScaler(),
            model_name)

        # Transform the three expression int a callable function




    # cross correlate the results

    #
