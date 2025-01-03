import torch
import torch.optim as optim
import torch.nn as nn
import model
import data
from torch.utils.data import DataLoader, random_split
import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchsummary

#define model
train_model = model.mushroom_model()

#hyper parameter
epochs = 100
batch_size = 16
loss_fn = nn.BCELoss()
optimizer = optim.Adam(train_model.parameters(), lr=1e-3)

# dataset
Data = data.Mushroom_data("Learn_AI\Binary_classification\mushroom_cleaned.csv")
data_size = len(Data)
train_data = int(data_size * 0.8)
test_data = data_size - train_data

train_dataset, test_dataset = random_split(Data, [train_data, test_data])

train_dataloader = DataLoader(train_dataset, batch_size, drop_last=True, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, drop_last=True, shuffle=True)

cnt = 0
total_loss = 0
# train
for epoch in range(1, epochs + 1) :
    train_model.train()
    cnt = 0
    total_loss = 0
    for x, y in tqdm.tqdm(train_dataloader) :
        optimizer.zero_grad()
        pred = train_model(x)

        pred = torch.flatten(pred)
        loss = loss_fn(pred, y)
        loss.backward()

        total_loss += loss.item()
        cnt += 1
        optimizer.step()
    print(f"avg loss : {total_loss/cnt}")

    # test
    test_cnt = 0
    test_loss = 0
    train_model.eval()
    for test_x, test_y in test_dataloader :
        test_pred = train_model(test_x)
        
        test_cnt += 1

        test_pred = torch.flatten(test_pred)
        test_loss += loss_fn(test_pred, test_y).item()
    print(f'test loss : {test_loss/test_cnt}')
