import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch.utils
import torch.utils.data

device = ('cuda' if torch.cuda.is_available() else 'cpu')

data_path = r'Learn_AI\Linear_classifi\fish_data.csv'

class fish_data(torch.utils.data.Dataset) :

    def __init__(self, path) -> None:
        df = pd.read_csv(path)
        self.species_num = {
            'Anabas_testudineus' : 0,
            'Coilia_dussumieri' : 1,
            'Otolithoides_biauritus' : 2,
            'Otolithoides_pama' : 3,
            'Pethia_conchonius' : 4,
            'Polynemus_paradiseus' : 5,
            'Puntius_lateristriga' : 6,
            'Setipinna_taty' : 7,
            'Sillaginopsis_panijus' : 8
        }
        self.label = np.array(list(df.iloc[:, 1:].values))
        self.label = np.array(map(lambda x: self.species_num[x], self.label))
        self.feature = np.array(df.iloc[:,1:].values, dtype=float)
        self.length = len(df)
        
    def __getitem__(self, index) :
        x = torch.LongTensor(self.label[index])
        y = torch.LongTensor(self.feature[index])
        return x, y
        
    def __len__(self) :
        return self.length
    
train_dataset = fish_data(data_path)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
print(train_dataloader.dataset.__getitem__(0))
