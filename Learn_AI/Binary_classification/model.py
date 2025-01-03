import torch
import torch.nn as nn

class mushroom_model(nn.Module) :
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(7, 4),
            nn.ReLU(),

            nn.Linear(4, 2),
            nn.ReLU(),

            nn.Linear(2, 1),
            nn.Sigmoid()
        ) 
    def forward(self, x) :
        return self.layer(x)
