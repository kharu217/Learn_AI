import torch
import glob
import cv2
import torch.utils
import torch.utils.data

import torch.utils.data.dataloader
import torchvision
import matplotlib.pyplot as plt
import numpy

import model

img_path = "C:\\Users\\User\\Pictures\\Image_datas\\DATA_DY"
label = torch.tensor([[0], [1], [0], [0], [0],
                      [1], [0], [0], [1], [1],
                      [1], [0], [1], [1], [0],
                      [1], [0], [0], [1], [1],
                      [1], [0], [1], [1], [0],
                      [1], [0], [1], [0], [1]])

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, labels, img_paths):
        self.img_paths = glob.glob(img_paths + '\\' + '*.jpg')
        self.lables = labels
        self.imgs = []

        for img_path in self.img_paths:
            img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
            img = img.transpose((2, 0, 1))
            self.imgs.append(img)

        self.imgs = numpy.array(self.imgs)
        self.lables = numpy.asarray(self.lables)
    def __len__(self):
        return len(self.lables)

    def __getitem__(self, idx):
        img_ = self.imgs[idx]
        label_ = self.lable[idx]
        return img_, label_

Dataset = CustomDataset(label, img_path)
Dataload = torch.utils.data.DataLoader(Dataset, batch_size=5)

model_t = model.DY_CNN()

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(params=model_t.parameters(), lr=1)
def train(dataloader, model, loss_fn, optimizer) :
    size = len(dataloader.dataset)
    
    for i, data in enumerate(dataloader) :
        img, label = data

        img = img.unsqueeze(0)
        pred = model(img)
        loss = loss_fn(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0 :
            loss, current = loss.item(), (i + 1) * len(img)
            print(f'loss : {loss:>7f} [{current:>5d}/{size:>5d}]')
            
    torch.save(model_t.state_dict(), 'SIMPLE_CNN\\dy_CNN.h5')

train(Dataload, model_t, loss_fn, optimizer)
