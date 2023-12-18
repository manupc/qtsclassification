import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

# Antsatz of RealAmplitudes with rotations on XYZ
def LayerRealAmplitudesXYZ(inputs, W):
    num_qubits= W.shape[0]
    for i in range(num_qubits):
        qml.Rot(W[i, 0], W[i, 1], W[i, 2], wires=i) 
    for i in range(num_qubits):
        qml.CNOT(wires= [i, (i+1)%num_qubits])



# Capa de agregación por suma
class SumAggregationLayer_v0(nn.Module):
    def __init__(self, size_in, size_out):
        super().__init__()
        num_components= int(np.ceil(size_in/size_out))
        idx= []
        current= []
        for v in range(size_in):
            current.append(v)
            
            if len(current)>=num_components:
                current= torch.LongTensor(current)
                idx.append(current)
                current= []
        if current:
            current= torch.LongTensor(current)
            idx.append(current)

        self.indices= idx

    def forward(self, x):
        outs= []
        for idx in self.indices:
            outs.append( x[:, idx].sum(dim=1).view(-1, 1) )
        return torch.concat(outs, dim=1)



# Capa de agregación por suma
class SumAggregationLayer_v2(nn.Module):
    def __init__(self, size_in, size_out):
        super().__init__()
        num_components= int(np.ceil(size_in/size_out))
        if size_in/num_components < size_out:
            num_components= int(np.floor(size_in/size_out))
        
        idx= []
        current= []
        k= 0
        
        currentComponents= num_components
        if size_in % size_out >k:
            currentComponents+= 1
        for v in range(size_in):
            current.append(v)
            
            if len(current)>=currentComponents:
                k+= 1
                current= torch.LongTensor(current)
                idx.append(current)
                current= []
                currentComponents= num_components
                if size_in % size_out >k:
                    currentComponents+= 1
        if current:
            current= torch.LongTensor(current)
            idx.append(current)

        self.indices= idx

    def forward(self, x):
        outs= []
        for idx in self.indices:
            outs.append( x[:, idx].sum(dim=1).view(-1, 1) )
        return torch.concat(outs, dim=1)





# Capa de agregación por suma
class SumAggregationLayer(nn.Module):
    def __init__(self, size_in, size_out):
        super().__init__()
        num_components= int(np.ceil(size_in/size_out))
        if size_in/num_components < size_out:
            num_components= int(np.floor(size_in/size_out))
        
        idx= []
        current= []
        k= 0
        currentComponents= num_components
        if size_in % num_components >k:
            currentComponents+= 1
        for v in range(size_in):
            current.append(v)
            
            if len(current)>=currentComponents:
                k+= 1
                current= torch.LongTensor(current)
                idx.append(current)
                current= []
                currentComponents= num_components
                if size_in % num_components >k:
                    currentComponents+= 1
        if current:
            current= torch.LongTensor(current)
            idx.append(current)

        self.indices= idx

    def forward(self, x):
        outs= []
        for idx in self.indices:
            outs.append( x[:, idx].sum(dim=1).view(-1, 1) )
        return torch.concat(outs, dim=1)



