import model
import torch

model_r = model.fish_classify()
model_r.load_state_dict(torch.load(r'C:\Users\User\Desktop\github\Learn_AI\Learn_AI\Linear_classifi\fish_classfi.h5'))

print(model_r(torch.Tensor([11.28,4.12,0.37])))
