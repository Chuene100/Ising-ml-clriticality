
# Deep Neural Network (DNN) model definition using PyTorch
import torch
import torch.nn as nn

inner_dim = 128

class VDNN(nn.Module):
    """Deep fully connected neural network for Ising model."""
    def __init__(self, nl=6):
        super(VDNN, self).__init__()
        layers = [nn.Linear(40*40, inner_dim), nn.ReLU()]
        for _ in range(nl-1):
            layers.append(nn.Linear(inner_dim, inner_dim))
            layers.append(nn.ReLU())
        layers.extend([nn.Linear(inner_dim, 1), nn.Sigmoid()])
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)

class VDNN_one(nn.Module):
    """Shallow fully connected neural network for Ising model."""
    def __init__(self, nl=1):
        super(VDNN_one, self).__init__()
        layers = [nn.Linear(40*40, inner_dim), nn.ReLU()]
        for _ in range(nl-1):
            layers.append(nn.Linear(inner_dim, inner_dim))
            layers.append(nn.ReLU())
        layers.extend([nn.Linear(inner_dim, 1), nn.Sigmoid()])
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)

class SimpleNet(nn.Module):                           # nn.Module is a subclass from which we inherit
    def __init__(self,nl=2):                                     # Here you define the structure
        super(SimpleNet, self).__init__()
        layers=[]
        layers.extend((nn.Linear(1,inner_dim), nn.ReLU(), nn.Linear(inner_dim,1), nn.Sigmoid()))
        self.layers = nn.Sequential(*layers)
    def forward(self,x):
        return self.layers(x)