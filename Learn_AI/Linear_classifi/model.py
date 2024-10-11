import torch
import torch.nn as nn

class fish_classify(torch.nn.Module) :
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(3, 12)
        self.layer2 = nn.Linear(12, 24)
        self.layer3 = nn.Linear(24, 9)
        self.relu = nn.ReLU()
    
    def forward(self, x) :
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        return x
