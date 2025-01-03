import torch
import torch.nn as nn
import torch.optim as optim

Embedd = nn.Embedding(10, 4)
print(Embedd.weight)
print('----------------')

input_t = torch.LongTensor([0,1,2])
print(Embedd(input_t))
