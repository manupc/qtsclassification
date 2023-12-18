import torch
import torch.nn as nn
import numpy as np
import time
import pickle
import copy

def accuracy(labels, predictions):

    loss = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            loss = loss + 1
    loss = loss / len(labels)
    return loss


def evaluateModelAccuracy(model, X, Y):
    with torch.no_grad():
        probs= model(X).cpu().numpy()

        if probs.shape[1] > 1: # Multi-class
            argprobs= np.argmax(probs, axis=1)
            acc = accuracy(Y.cpu().numpy(), argprobs)
        
        else: # Binary classification
            argprobs= np.where(probs>0, 1.0, 0.0)
            acc = accuracy(Y.cpu().numpy(), argprobs)
            
    return acc


def summaryResults(trAccList, tsAccList, times, verbose=True):
    meanTr, meanTs= np.mean(trAccList), np.mean(tsAccList)
    sdTr, sdTs= np.std(trAccList), np.std(tsAccList)
    maxTr, maxTs= np.max(trAccList), np.max(tsAccList)
    minTr, minTs= np.min(trAccList), np.min(tsAccList)
    tsAcc= np.array(tsAccList).squeeze()
    trAcc= np.array(trAccList).squeeze()
    bestTestAcc= np.max(tsAcc)
    iBestTest= tsAcc>=bestTestAcc
    tsAcc= tsAcc[iBestTest]
    trAcc= trAcc[iBestTest]
    iBest= np.argmax(trAcc)
    bestTr, bestTs= trAcc[iBest], tsAcc[iBest]
    meanT, sdT= np.mean(times), np.std(times)

    if verbose:
        print('Summary of results for {} executions: '.format(len(trAccList)))
        print('"""""""""""""""""""""""""""""""""""""""""""""""')
        print('Avg Execution time: {:0.2f} +- {:0.2f} '.format(meanT, sdT))
        print('Train Acc: {:0.2f} +- {:0.2f} | Best: {:0.2f} | Worst: {:0.2f}'.format(meanTr*100, sdTr*100, maxTr*100, minTr*100))
        print('Test Acc: {:0.2f} +- {:0.2f} | Best: {:0.2f} | Worst: {:0.2f}'.format(meanTs*100, sdTs*100, maxTs*100, minTs*100))
        print('Best execution Acc: {:0.2f} train / {:0.2f} test'.format(bestTr*100, bestTs*100))
        
    return meanTr, sdTr, maxTr, minTr, meanTs, sdTs, maxTs, minTs, iBest

def summaryResultsFromFile(resultsFile):
    try:
        with open(resultsFile, 'rb') as handle:
            storedData = pickle.load(handle)
        allTimes= storedData[1]
        allTrAcc= storedData[2]
        allTsAcc= storedData[3]
        summaryResults(allTrAcc, allTsAcc, allTimes, verbose=True)

    except:
        print('Cannor read file {}'.format(resultsFile))

def loadResultsFile(resultsFile):
    try:
        with open(resultsFile, 'rb') as handle:
            storedData = pickle.load(handle)
        AllLossHistories= storedData[0]
        allTimes= storedData[1]
        allTrAcc= storedData[2]
        allTsAcc= storedData[3]
        allModels= storedData[4]

    except:
        AllLossHistories= []
        allTimes= []
        allTrAcc= []
        allTsAcc= []
        allModels= []
    return AllLossHistories, allTimes, allTrAcc, allTsAcc, allModels


def saveResultsFile(resultsFile, AllLossHistories, allTimes, allTrAcc, allTsAcc, allModels):
    storedData= (AllLossHistories, allTimes, allTrAcc, allTsAcc, allModels)
    with open(resultsFile, 'wb') as handle:
        pickle.dump(storedData, handle, protocol=pickle.HIGHEST_PROTOCOL)



def Train(model, optimizer, lossF, MaxIter, xTr, yTr, xTs, yTs, 
          testIterations, accThreshold, verbose=False):
    
    t0= time.time()
    lossHistory= []
    lastTestIt= None
    lastAcc_val= None
    bestModel= None
    bestTrAcc= 0.0
    bestTsAcc= 0.0
    for it in range(MaxIter):

        # Update the weights by one optimizer step
        #batch_index = np.random.randint(0, num_train, (batch_size,))
        optimizer.zero_grad()
        probs= model(xTr)

        # Calculate cross entropy
        loss_evaluated = lossF(probs, yTr)
        loss_evaluated.backward()
        optimizer.step()
        lossHistory.append(loss_evaluated.item())

        # Test Accuracy on training and test
        acc_train = evaluateModelAccuracy(model, xTr, yTr)
        if (acc_train >= bestTrAcc and testIterations<=0):
            bestTrAcc= acc_train
            bestModel= copy.deepcopy(model)
        
        if ((testIterations>0) and (it % testIterations == 0)) or (it>=MaxIter-1):
            acc_val = evaluateModelAccuracy(model, xTs, yTs)
            lastTestIt= it
            lastAcc_val= acc_val
            
            if (acc_val > bestTsAcc) or (acc_val >= bestTsAcc and acc_train>=bestTrAcc):
                bestTrAcc= acc_train
                bestTsAcc= acc_val
                bestModel= copy.deepcopy(model)
                
            

        if verbose:
            iterCos= "Iter: {:5d} | Cost: {:0.7f}".format(it+1, loss_evaluated.item())
            train= " | Acc train: {:0.2f}".format(acc_train*100)
            if lastAcc_val is None:
                test= ""
            else:
                test= " | Acc validation: {:0.2f} (it {:5d})".format(lastAcc_val*100, lastTestIt+1)
            t= " | Time: {:0.2f}s.".format(time.time()-t0)
            

            print(iterCos+train+test+t)

        if acc_train >= accThreshold:
            if lastAcc_val is None or (lastAcc_val>=accThreshold and it==lastTestIt):
                break
    tf= time.time()
    return bestModel, lossHistory, tf-t0
