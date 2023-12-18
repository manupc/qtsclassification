import torch
from scipy.io import arff
import pandas as pd
import numpy as np
from tslearn.preprocessing import TimeSeriesResampler

# Class to model and manage datasets
class Dataset:
    def __init__(self, name, path, trainFile, testFile):
        
        # Dataset name
        self.name= name

        # Path to data
        self.path= path

        # Load training and test data
        xTr, yTr= self.__loadFile(path+'/'+trainFile)
        xTs, yTs= self.__loadFile(path+'/'+testFile)
        
        # Make labels ordinals starting from 0
        trueLabels= np.sort(np.unique(np.concatenate((yTr, yTs))))
        for i, v in enumerate(trueLabels):
            yTr[yTr ==v]= i
            yTs[yTs ==v]= i
        self.labels= np.sort(np.unique(np.concatenate((yTr, yTs))))
        
        # Save dataset
        self.xTr= xTr
        self.yTr= yTr
        self.xTs= xTs
        self.yTs= yTs
        self.labels= np.sort(np.unique(np.concatenate((yTr, yTs))))
        self.originalLength= self.xTr.shape[1]


    def __loadFile(self, file):
        data = arff.loadarff(file)
        data = pd.DataFrame(data[0]).to_numpy()
        train= data[:, :-1].astype(np.float32)
        test= data[:, -1].astype(int)
        return train, test


    def numLabels(self):
        return len(self.labels)
    
    def TimeSeriesLength(self):
        return self.xTr.shape[1]

    def numTrainingSamples(self):
        return self.xTr.shape[0]

    def numTestSamples(self):
        return self.xTs.shape[0]
    
    def TimeSeriesReshape(self, newlength):
        self.xTr= TimeSeriesResampler(sz=newlength).fit_transform(self.xTr).squeeze()
        self.xTs= TimeSeriesResampler(sz=newlength).fit_transform(self.xTs).squeeze()

    def trainingPatterns(self):
        return self.xTr, self.yTr

    def shuffle(self):
        p= np.random.permutation(self.xTr.shape[0])
        self.xTr= self.xTr[p]
        self.yTr= self.yTr[p]

    def diff(self):
        self.xTr= self.xTr[:, 1:] - self.xTr[:, :-1]
        self.xTs= self.xTs[:, 1:] - self.xTs[:, :-1]

    def toOneHotLabels(self):
        if len(self.yTr.shape)>1:
            if self.yTr.shape[1] == len(self.labels):
                return

        yTr= np.zeros((self.yTr.shape[0], len(self.labels)), dtype=int)
        for v in self.labels:
            yTr[self.yTr==v, v]= 1
        self.yTr= yTr

        yTs= np.zeros((self.yTs.shape[0], len(self.labels)), dtype=int)
        for v in self.labels:
            yTs[self.yTs==v, v]= 1
        self.yTs= yTs



    def toCardinalLabels(self):
        if (len(self.yTr.shape)==1) or (len(self.yTr.shape)==2 and self.yTr.shape[1]==1):
            return
            
        yTr= np.empty(self.yTr.shape[0], dtype=int)
        for v in self.labels:
            yTr[self.yTr[:, v]==1]= v
        self.yTr= yTr

        yTs= np.empty(self.yTs.shape[0], dtype=int)
        for v in self.labels:
            yTs[self.yTs[:, v]==1]= v
        self.yTs= yTs



    def __str__(self):
        text= '=========================='+'\n'
        text+= 'Dataset '+self.name+'\n'
        text+= '-------------------------'+'\n'
        text+= '#Training patterns: '+str(self.xTr.shape[0])+'\n'
        text+= '#Test patterns: '+str(self.xTs.shape[0])+'\n'
        text+= '#classes: '+str(len(self.labels))+' ('+str(self.labels.tolist())+')'+'\n'
        text+= '# of instances for each class:\n'
        for i, l in enumerate(self.labels):
            text+= '\t class '+str(l)+': '+str( np.sum(self.yTr==l) )+'\n'
        text+= 'Time series length (original): '+str(self.originalLength)+'\n'
        if self.originalLength != self.xTr.shape[1]:
            text+= '    Reshaped to: '+str(self.xTr.shape[1])+'\n'
        text+= 'Mean value of time series: '+str(np.mean(self.xTr))+'\n'
        text+= 'Required qbits: '+str(self.requiredQubits())+'\n'
        text+= '=========================='+'\n'
        return text

    def getName(self):
        return self.name


    def toTensor(self):
        type=0
        if len(self.yTr.shape)>1:
            if self.yTr.shape[1] == len(self.labels):
                type= 1
        self.xTr= torch.FloatTensor(self.xTr)
        self.xTs= torch.FloatTensor(self.xTs)
        if type==0:
            self.yTr= torch.LongTensor(self.yTr)
            self.yTs= torch.LongTensor(self.yTs)
        else:
            self.yTr= torch.FloatTensor(self.yTr)
            self.yTs= torch.FloatTensor(self.yTs)

    def getTrainingData(self):
        return self.xTr, self.yTr

    def getTestData(self):
        return self.xTs, self.yTs

    def requiredQubits(self):
        num_qubits= np.log2(self.xTr.shape[1])
        if num_qubits.is_integer():
            return int(num_qubits)
        else:
            return int(num_qubits)+1
