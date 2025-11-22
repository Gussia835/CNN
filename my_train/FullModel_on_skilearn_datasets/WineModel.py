import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_wine




class WineDataset(data.Dataset):
    def __init__(self):
        self.data = _global_var_data_x # тензор размерностью (178, 13), тип float32
        self.target = _global_var_target # тензор размерностью (178, ), тип int64 (long)

        self.length = len(self.target)
        self.categories = ['class_0', 'class_1', 'class_2'] # названия классов

    def __getitem__(self, item):
        return self.data[item], self.target[item]

    def __len__(self):
        return self.length

class WineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(in_features=13,
                                 out_features=32)
        self.linear2 = nn.Linear(in_features=32,
                                 out_features=16)
        self.linear3 = nn.Linear(in_features=16,
                                 out_features=3)
    
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x

epochs = 20
batch_size = 16
lr = 0.01

d_train = WineDataset()
train_data = data.DataLoader(d_train,
                             batch_size=batch_size,
                             shuffle=True)

model = WineModel()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=lr)
loss_func = nn.CrossEntropyLoss()

model.train()
for _e in range(epochs):
    for x_train, y_train in train_data:
        y_pred = model(x_train)
        error = loss_func(y_pred, y_train)
        
        optimizer.zero_grad()
        error.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    all_data = d_train.data
    predictions = model(all_data)
    predicted_classes = predictions.argmax(dim=1)
    correct = (predicted_classes == d_train.target)
    Q = correct.float().mean().item()