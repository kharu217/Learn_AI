import pandas as pd
import torch

data = pd.read_csv(r'C:\Users\User\Desktop\github\Learn_AI\Learn_AI\Linear_classifi\fish_data.csv')
species = data.loc[:, ('species')]
feature = torch.from_numpy(data.loc[:, ['length', 'weight', 'w_l_ratio']].values)

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

species = torch.tensor(list(map(lambda x : species_num[x], species.values)))

print(feature)
print(species)
