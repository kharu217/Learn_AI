import torch
import model

model_t = model.MUL_class()

data_l = []
label = []

for w in range(1, 10) :
    for h in range(1, 10) :
        data_l.append([w, h])
        label.append([w * h])

data_l = torch.FloatTensor(data_l)
label = torch.FloatTensor(label)

optimizer = torch.optim.Adam(model_t.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

model_t.train()
for epoch in range(0, 300) :
    for i in range(0, len(data_l)) :
        optimizer.zero_grad()

        pred = model_t(data_l[i])
        loss = loss_fn(pred, label[i])

        loss.backward()
        optimizer.step()
    print(epoch)
    print(loss.item())
torch.save(model_t.state_dict(), r'Learn_AI\MUL_AI\model_save\mul_model.h5')