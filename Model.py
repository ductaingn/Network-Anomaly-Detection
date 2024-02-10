import torch
import torch.nn as nn
import torch.nn.functional as func

class Model(nn.Module):
    def __init__(self,in_features=31, h1=32, h2=16, out_features=5):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self,x):
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.out(x)

        return x