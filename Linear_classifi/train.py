import torch
import model
import data

epochs = 200

model_t = model.fish_classify()
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_t.parameters(), lr=1e-3)

model_t.train()
for i in range(1, epochs + 1) :
    for d in range(0, len(data.feature)) :
        pred = model_t(data.feature[d])
        losses = loss(pred, data.species_n[d])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'{i} epoch : \n {loss.item()}')

torch.save(model_t.state_dict(), r'Linear_classifi\fish_classfi.h5')