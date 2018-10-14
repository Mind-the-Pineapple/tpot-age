#!/usr/bin/env python

import os
import argparse
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LinearRegression

from BayOptPy.helperfunctions import get_data

parser = argparse.ArgumentParser()
parser.add_argument('-nogui',
                    dest='nogui',
                    action='store_true',
                    help='No gui')
args = parser.parse_args()

if __name__ == '__main__':

    print('The current args are: %s' %args)
    #project_wd = '/BayOpt'
    project_wd = os.getcwd()
    project_sink, demographics, imgs, maskedData = get_data(project_wd)

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
