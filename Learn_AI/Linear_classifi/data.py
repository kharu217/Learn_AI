import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

device = ('cuda' if torch.cuda.is_available() else 'cpu')

species_num = {
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

class fish_data(Dataset) :
    def __init__(self, data_path) :
        data = pd.read_csv(data_path)
        self.species = data.loc[:, ('species')].values
        self.species = torch.tensor(list(map(lambda x : species_num[x] ,self.species)), dtype=torch.float32)
        self.species = torch.nn.functional.one_hot(self.species.to(torch.int64), 9)
        self.feature = torch.from_numpy(data.loc[:, ['length', 'weight', 'w_l_ratio']].values).float()
        
    def __len__(self) :
        return len(self.feature)

    def __getitem__(self, index):
        return self.feature[index] ,self.species[index]


