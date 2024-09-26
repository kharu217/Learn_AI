import torch
from model import sub_model

t_model = sub_model()
t_model.load_state_dict(torch.load('SUB_AI\\test_model.pth'))

t_model.eval()

print(t_model(torch.FloatTensor([3, 26])))
