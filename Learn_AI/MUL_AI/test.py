import torch
import model

model_s = model.MUL_class()
model_s.load_state_dict(torch.load(r'Learn_AI\MUL_AI\model_save\mul_model.h5'))

print(model_s(torch.FloatTensor([12, 12])))
