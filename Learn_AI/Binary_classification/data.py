import torch
from torch.utils.data import DataLoader, Dataset
import pandas
import numpy
import torch.nn.functional as F

class Mushroom_data(Dataset) :
    def __init__(self, f_path):
        super().__init__()
        raw_data = pandas.read_csv(f_path)
        self.x = F.normalize(torch.FloatTensor(raw_data.iloc[:, :8].values), dim=0)
        self.y = torch.FloatTensor(raw_data.iloc[:, -1].values)
    
    def __len__(self) :
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

if __name__ == "__main__" :
    test_data = Mushroom_data("Learn_AI\Binary_classification\mushroom_cleaned.csv")
    print(test_data.x[:10])
    print(test_data.y[:10])
