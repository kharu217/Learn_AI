import torch
import model
import data
import numpy
import tqdm

epochs = 10

dataset = data.fish_data(r'C:\Users\User\Desktop\github\Learn_AI\Learn_AI\Linear_classifi\fish_data.csv')
dataloader = data.DataLoader(dataset, shuffle=True)

model_t = model.fish_classify()
loss = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model_t.parameters(), lr=1e-3)

for i in range(1, epochs) :
    model_t.train()
    epoch_loss = 0
    for feature, label in tqdm.tqdm(dataloader):
        optimizer.zero_grad()

        pred = model_t(feature).transpose(1, 0).squeeze(dim=1)
        label = label.transpose(1, 0).squeeze(dim=1)

        losses = loss(pred, label)
        losses.backward()
        optimizer.step()
        epoch_loss += losses.item()
    print(f'{i} epoch : \n {epoch_loss / len(dataloader)}')
    # 각 layer의 이름과 파라미터 출력
    # for name, child in model_t.named_children():
    #     for param in child.parameters():
    #         print(name, param)


torch.save(model_t.state_dict(), r'C:\Users\User\Desktop\github\Learn_AI\Learn_AI\Linear_classifi\fish_classfi.h5')
