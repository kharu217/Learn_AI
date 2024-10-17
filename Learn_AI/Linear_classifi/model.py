import torch
import torch.nn as nn

class fish_classify(torch.nn.Module) :
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(3,64)
        self.relu1 = nn.ReLU()

        self.dropout1 = nn.Dropout(0.2)

        self.layer2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()

        self.dropout2 = nn.Dropout1d(0.2)

        self.layer3 = nn.Linear(32, 16)
        self.relu3 = nn.ReLU()

        self.layer4 = nn.Linear(16, 9)
        self.sfmax = nn.Softmax()

    
    def forward(self, x) :
        out = self.layer1(x)
        out = self.relu1(out)

        out = self.layer2(out)
        out = self.relu2(out)

        out = self.layer3(out)
        out = self.relu3(out)

        out = self.layer4(out)
        out = self.sfmax(out)

        return out
