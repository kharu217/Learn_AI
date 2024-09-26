import torch
import numpy

class DY_CNN(torch.nn.Module) :
    def __init__(self):
        super(DY_CNN, self).__init__()
        self.layer1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.layer2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)

        self.linear1 = torch.nn.Linear(in_features=339, out_features=600)
        self.linear2 = torch.nn.Linear(in_features=600, out_features=120)
        self.linear3 = torch.nn.Linear(in_features=120, out_features=24)
        self.linear4 = torch.nn.Linear(in_features=24, out_features=1)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x) :
        output = self.relu(self.layer1(x))
        output = self.relu(self.layer2(x))
        output = output.view(output.size(0), -1)
        output = self.linear1(output)
        output = self.linear2(output)
        output = self.linear3(output)
        return output