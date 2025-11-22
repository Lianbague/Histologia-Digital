import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F


class NeuralNetwork(nn.Module): 
    def __init__(self, in_features, out_features): 
        super().__init__() 
              
        self.net = nn.Sequential(
            nn.Linear(in_features, int(in_features / 2)),  
            nn.ReLU(inplace=True), 
            nn.Dropout(p=0.5), 
            nn.Linear(int(in_features / 2), out_features),
            nn.Sigmoid() # important clasificacio binaria
        )
        
    def forward(self, x): 
        return self.net(x)


class Attention(nn.Module):
    def __init__(self,in_features, decom_space, attention_branches):
        super(Attention, self).__init__() 
        self.M = in_features
        self.L = decom_space
        self.ATTENTION_BRANCHES = attention_branches


        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh(),
            nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
        )

    def forward(self, x):

        # H feature vector matrix  # NV vectors x M dimensions
        H = x.squeeze(0)
        # Attention weights
        A = self.attention(H)  # NVxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxNV
        A = F.softmax(A, dim=1)  # softmax over NV
        
        # Context Vector (Attention Aggregation)
        Z = torch.mm(A, H)  # ATTENTION_BRANCHESxM 
        
        return Z, A

# Gated Attention is a more advanced attention mechanism that combines two different transformations of the input to compute attention weights.
class GatedAttention(nn.Module):
    def __init__(self,in_features, decom_space, attention_branches):
        super(GatedAttention, self).__init__()
        self.M = in_features
        self.L = decom_space
        self.ATTENTION_BRANCHES = attention_branches
        
        # Matrix for Query decomposition
        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh()
        )
        # Matrix for Keys decomposition
        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix U
            nn.Sigmoid()
        )

        self.attention_w = nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)


    def forward(self, x):
        # H feature vector matrix  # NV vectors x M dimensions
        H = x.squeeze(0)
        ## Self Attention weights
        # Input Vector Query Decomposition, Q
        A_V = self.attention_V(H)  # NVxL (Projecion of the V input vectors into L dim space)
        # Input Vector Keys Decomposition, K
        A_U = self.attention_U(H)  # NVxL
        # Attention Matrix from Product Q*K 
        A = self.attention_w(A_V * A_U) # element wise multiplication # NVxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxNV
        A = F.softmax(A, dim=1)  # softmax over NV dimension
        
        ## Context Vector (Attention Aggregation)
        Z = torch.mm(A, H)  # ATTENTION_BRANCHESxM

        return Z, A


class NeuralNetwork_withAttention(nn.Module): 
    def __init__(self, input_dim, project_dim=512, decom_space=128, attention_branches=1, attention_type='GatedAttention'): 
        super().__init__() 
        
        # SOLUCIO CUDA OUT OF MEMORY
        # Capa de projeccio: redueix la dimensio gegant que teniem
        # Redueix de 32768 -> 512 
        self.projector = nn.Sequential(
            nn.Linear(input_dim, project_dim),
            nn.ReLU(),
            nn.Dropout(0.25)
        )

        # Initialitza attention mechanism
        if attention_type == 'attention':
            self.attention = Attention(project_dim, decom_space, attention_branches)
        elif attention_type == 'GatedAttention':
            self.attention = GatedAttention(project_dim, decom_space, attention_branches)
        else:
            raise ValueError("Invalid attention type. Choose 'attention' or 'GatedAttention'.")

        self.NeuralNetwork = NeuralNetwork(project_dim * attention_branches, 1)

    def forward(self, x): 
        # Reduir dimensionalitat
        x = x.squeeze(0) # [N, 32768]
        H = self.projector(x) # [N, 512] -> Aqui ahorramos gigabytes de memoria
        H = H.unsqueeze(0) # [1, N, 512]
        
        Z,A = self.attention(H)  # Z: ATTENTION_BRANCHESxM
        Z = Z.view(-1, Z.shape[0]*Z.shape[1])  # This flattens Z to shape [1, ATTENTION_BRANCHES*M] this is because the NN input layer expects that shape
        output = self.NeuralNetwork(Z)  # output: ATTENTION_BRANCHESxout
        return output, A