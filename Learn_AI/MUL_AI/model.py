import torch

class MUL_class(torch.nn.Module) :
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.layer1 = torch.nn.Linear(2, 32)
        self.layer2 = torch.nn.Linear(32, 64)
        self.layer3 = torch.nn.Linear(64, 16)
        self.layer4 = torch.nn.Linear(16, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, input_) :
        input_ = self.relu(self.layer1(input_))
        input_ = self.relu(self.layer2(input_))
        torch.nn.Dropout(0.2)
        input_ = self.relu(self.layer3(input_))
        input_ = self.relu(self.layer4(input_))
        return input_
