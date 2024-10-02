import torch

class MUL_class(torch.nn.Module) :
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.layer1 = torch.nn.Linear(2, 32)
        self.layer2 = torch.nn.Linear(32, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, input_) :
        input_ = self.relu(self.layer1(input_))
        input_ = self.relu(self.layer2(input_))
        return input_
