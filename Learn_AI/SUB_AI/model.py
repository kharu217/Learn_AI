import torch

class sub_model(torch.nn.Module) :
    def __init__(self) :
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 1)
        self.relu = torch.nn.ReLU()
        
    def forward(self, x) :
        return self.relu(self.linear1(x))