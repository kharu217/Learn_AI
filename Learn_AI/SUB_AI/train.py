import torch
from SUB_AI.model import sub_model

x_data = [[10, 5],[69, 19],[6, 2],[2,1],[53, 10],[4,3], [6,5], [1000, 121], [10,9], [8, 7]]
y_data = [[5], [50], [4], [1], [43], [1], [1], [879], [1], [1]]

x_data = torch.FloatTensor(x_data)
y_data = torch.FloatTensor(y_data)

model = sub_model()

optimizer = torch.optim.Adam(model.parameters(), lr=1)

model.train()

epochs = 1000
for epoch in range(epochs + 1) :
    pred = model(x_data)
    
    loss = torch.nn.functional.mse_loss(pred, y_data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0 :
        print(f"{epoch} : loss : {loss.item()}\n")
        print(model.state_dict())
torch.save(model.state_dict(), 'test_model.pth')
