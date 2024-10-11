import torch
import model
import data

epochs = 100

model_t = model.fish_classify()
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_t.parameters(), lr=1)

model_t.train()
for i in range(1, epochs) :
    for label, fetr in data.train_dataloader :
        
        print(label, fetr)
        
        label = label.to(data.device)
        fetr = fetr.to(data.device)
        
        optimizer.zero_grad()
        
        pred = model_t(fetr)
        losses = loss(pred, label)
        
        losses.backward()
        optimizer.step()
    print(f'{i} epoch : \n {losses.item()}')

torch.save(model_t.state_dict(), r'Linear_classifi\fish_classfi.h5')