import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

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

data_path = r'Learn_AI\Linear_classifi\fish_data.csv'

species = pd.read_csv(data_path, usecols=['species'])
feature = pd.read_csv(data_path, usecols=['length', 'weight', 'w_l_ratio'])
feature = torch.tensor(feature.values).float()
species_n = []

print(feature)

print(len(species))
for i in range(1, len(species)) :
    species_n.append(species_num[species['species'][i]])
species_n = torch.tensor(species_n)
species_n = F.one_hot(species_n, num_classes=9).float()
print(species_n)


