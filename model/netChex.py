import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from DensenetModels import DenseNet121

class netChex(nn.Module):
    
    def __init__(self, params):
        
        super(Net, self).__init__()
        self.num_channels = params.num_channels
        
    def forward(self, s):
        
    