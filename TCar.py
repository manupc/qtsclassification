import pennylane as qml
import numpy as np
from Dataset import Dataset
import torch
import torch.nn as nn
import QTools
from TrainingTools import *
import random

RSeed= 12345
np.random.seed(RSeed)
random.seed(RSeed)
torch.random.manual_seed(RSeed)


dataset= Dataset('CAR', './datasets', 'Car_TRAIN.arff', 'Car_TEST.arff')
resultsFile= 'TCar_R.pkl' # None for not saving results data

dataset.shuffle()
#dataset.diff()
#dataset.TimeSeriesReshape( 256 )
print(dataset)


if torch.cuda.is_available():
    print('------ CUDA IS AVAILABLE ---- ')
else:
    print('------ CUDA NOT AVAILABLE ---- ')
device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device= torch.device("cpu")
print('----- USING DEVICE: ', device)


MaxIter= 1000
MaxExper= 30
lossF= nn.CrossEntropyLoss()
accThreshold= 1.0
dataset.toTensor()
xTr, yTr= dataset.getTrainingData()
xTs, yTs= dataset.getTestData()
xTr, yTr, xTs, yTs= xTr.to(device), yTr.to(device), xTs.to(device), yTs.to(device)





# Load results data
if resultsFile is not None:
    AllLossHistories, allTimes, allTrAcc, allTsAcc, allModels= loadResultsFile(resultsFile)
    if len(allTrAcc)>=2: 
        summaryResults(allTrAcc, allTsAcc, allTimes, verbose=True)
else:
    AllLossHistories, allTimes, allTrAcc, allTsAcc, allModels= [], [], [], [], []

# Experiments
for exper in range(MaxExper):

    if exper >=len(allModels):

        np.random.seed(RSeed+exper)
        random.seed(RSeed+exper)
        torch.random.manual_seed(RSeed+exper)

        print('Dataset {}. Experiment {}'.format(dataset.getName(), exper+1))
        print('======================================')

        ###############################################
        # Create new model
        model = torch.nn.Sequential(
                                    nn.Linear(dataset.TimeSeriesLength(), 500),
                                    nn.ReLU(),
                                    nn.Linear(500, 16),
                                    nn.ReLU(),
                                    nn.Linear(16, dataset.numLabels()),
                                    nn.Softmax(dim=1)
        )
        model= model.to(device)

        # Training Params
        learning_rate= 0.001
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # END Create new model
        ###############################################


        model, lossHistory, time= Train(model= model, 
                                        optimizer= opt, 
                                        lossF= lossF,
                                        MaxIter= MaxIter,
                                            xTr= xTr, yTr= yTr, 
                                            xTs= xTs, yTs= yTs, 
                                            testIterations= 1,
                                            accThreshold= accThreshold,
                                            verbose= True)

        AllLossHistories.append(lossHistory)
        allTimes.append(time)
        allTrAcc.append( evaluateModelAccuracy(model, xTr, yTr ) )
        allTsAcc.append( evaluateModelAccuracy(model, xTs, yTs ) )
        allModels.append(model)


        # Save results to file
        if resultsFile is not None:
            saveResultsFile(resultsFile, AllLossHistories, allTimes, allTrAcc, allTsAcc, allModels)

summaryResults(allTrAcc, allTsAcc, allTimes, verbose=True)