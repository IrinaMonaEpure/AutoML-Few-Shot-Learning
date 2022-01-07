####################################################################
# Do NOT change this file
# Written by: Mike Huisman
####################################################################

import torch
import torch.nn as nn

from collections import OrderedDict

class SineNetwork(nn.Module):
    """
    Base-learner neural network for the sine wave regression task.
    
    """

    def __init__(self, criterion=nn.MSELoss(), in_dim=1, out_dim=1, zero_bias=True):
        """Initializes the model
        
        Parameters
        ----------
        criterion : nn.loss_fn
            Loss function to use
        in_dim : int
            Dimensionality of the input
        out_dim : int
            Dimensionality of the output
        zero_bias : bool, optional
            Whether to initialize biases of linear layers with zeros
            (default is Uniform(-sqrt(k), +sqrt(k)), where 
            k = 1/num_in_features)
        **kwargs : dict, optional
            Trash can for additional arguments. Keep this for constructor call uniformity
        """
        
        super().__init__()
        self.relu = nn.ReLU()
        
        self.model = nn.ModuleDict({'features': nn.Sequential(OrderedDict([
            ('lin1', nn.Linear(in_dim, 64)),
            ('tanh1', nn.Tanh()),
            ('lin2', nn.Linear(64, 64)),
            ('tanh2', nn.Tanh())]))
        })
        
        # Output layer
        self.model.update({"out": nn.Linear(64, out_dim)})
        self.criterion = criterion
        
        if zero_bias:
            for m in self.model.modules():
                if isinstance(m, nn.Linear):
                    m.bias = nn.Parameter(torch.zeros(m.bias.size()))

    def forward(self, x):
        """Feedforward pass of the network

        Take inputs x, and compute the network's output using its weights
        w1, w2, and w3

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor with shape (Batch size, 1)

        Returns
        ----------
        tensor
            Predictions with shape (Batch size, 1) of the network on inputs x 
        """

        features = self.model.features(x)
        out = self.model.out(features)
        return out