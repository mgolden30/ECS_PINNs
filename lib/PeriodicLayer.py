import numpy as np
import torch

from torch import nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PeriodicLayer(nn.Module):
    def __init__(self):
        #This will have no internal parameters
        super().__init__()
    
    def forward(self, x):
        '''
        INPUT: 
        x - [b,3] tensor where b is batch size and 3 are the spatiotemporal coordinates (x,y,t)
        '''

        s = torch.sin(x)
        c = torch.cos(x)

        trig = torch.cat( (s, c), dim=1  )
        return trig