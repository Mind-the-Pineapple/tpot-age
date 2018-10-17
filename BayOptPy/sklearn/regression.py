#!/usr/bin/env python

import os
import argparse
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LinearRegression

from BayOptPy.helperfunctions import get_data, get_paths

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
args = parser.parse_args()

if __name__ == '__main__':

    print('The current args are: %s' %args)

    project_wd, project_data, project_sink = get_paths(args.debug)
    demographics, imgs, maskedData = get_data(project_data)

    print('Running regression analyis with sklearn')


    # split train-test dataset
    targetAttribute = demographics['Age']
    Xtrain, Xtest, Ytrain, Ytest = model_selection.train_test_split(maskedData, targetAttribute, test_size=.4, random_state=42)
    print('Divided dataset into test and training')
    print('Check train test split sizes')
    print('X_train: ' + str(Xtrain.shape))
    print('X_test: '  + str(Xtest.shape))
    print('Y_train: ' + str(Ytrain.shape))
    print('Y_test: '  + str(Ytest.shape))

    # Do simple Linear regression
    print('Start simple linear regression')
    linReg = LinearRegression()
    linReg.fit(Xtrain, Ytrain)
    print('Performed simple linear regresssion')
    Ypred = linReg.predict(Xtest)

    # plot the results
    plt.scatter(Ytest, Ypred)
    plt.xlabel("Age: $Y_i$")
    plt.ylabel("Predicted Age: $\hat{Y}_i$")
    plt.title("Age vs Predicted Age: $Y_i$ vs $\hat{Y}_i$")
    if args.nogui:
        plt.savefig(os.path.join(project_sink, 'regression.png'))
    else:
        plt.show()
    print('Done')
