import torch
import model
import data

epochs = 200

dataset = data.fish_data(r'C:\Users\User\Desktop\github\Learn_AI\Learn_AI\Linear_classifi\fish_data.csv')
dataloader = data.DataLoader(dataset, shuffle=True)

model_t = model.fish_classify()
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_t.parameters(), lr=1e-1)

for i in range(1, epochs) :
    model_t.train()
    for feature, label in dataloader:
        optimizer.zero_grad()

        pred = model_t(feature)

        losses = loss(pred, label.to(torch.float16))
        losses.backward()
        optimizer.step()
    print(f'feature : {feature}')
    print(f'pred : {pred} \n label : {label}')
    print(f'{i} epoch : \n {losses.item()}')
    # 각 layer의 이름과 파라미터 출력
    # for name, child in model_t.named_children():
    #     for param in child.parameters():
    #         print(name, param)


torch.save(model_t.state_dict(), r'C:\Users\User\Desktop\github\Learn_AI\Learn_AI\Linear_classifi\fish_classfi.h5')
