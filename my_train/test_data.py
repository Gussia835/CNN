import torch
import torch.nn as nn
import torch.utils.data as data


def f(x):
    part_1 = 2 * torch.exp(-x / 2) 
    part_2 = 0.2 * torch.sin(x / 10)
    return part_1 + part_2 - 5


class FuncDataSet(data.Dataset):
    def __init__(self):
        self.coord_x = torch.arange(-5, 5, 0.1) 
        self.targets = f(self.coord_x)
        self.length = len(self.coord_x)

    def __getitem__(self, item):
        return self.coord_x[item], self.targets[item]

    def __len__(self):
        return self.length


d_train = FuncDataSet()
print(d_train[13])