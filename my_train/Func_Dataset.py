import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim


class FuncDataset(data.Dataset):
    def __init__(self):
        _range = torch.arange(-3, 3, 0.1)
        self.data = torch.tensor([(_x, _y) for _x in _range for _y in _range])
        self.target = self._func(self.data)
        self.length = len(self.target)  # размер обучающей выборки

    @staticmethod
    def _func(coord):
        _x, _y = coord[:, 0], coord[:, 1]
        return torch.sin(2 * _x) * torch.cos(3 * _y) + 0.2 * torch.cos(10 * _x) * torch.sin(8 * _x) + 0.1 * _x ** 2 + 0.1 * _y ** 2

    def __getitem__(self, item):
        return self.data[item], self.target[item]

    def __len__(self):
        return self.length


class FuncModel(nn.Module):
    def __init__(self):
        super().__init__()
        # модель однослойной полносвязной нейронной сети:
        self.layer1 = nn.Linear(in_features=6, 
                                out_features=1)

    def forward(self, coord):
        x, y = coord[:, 0], coord[:, 1]
        x.unsqueeze_(-1)
        y.unsqueeze_(-1)

        xx = torch.empty(coord.size(0), 6)
        xx[:, :3] = torch.cat([x, x ** 2, x ** 3], dim=1)
        xx[:, 3:] = torch.cat([y, y ** 2, y ** 3], dim=1)
        y = self.layer1(xx)
        return y


model = FuncModel()
epochs = 20
batch_size = 16

d_train = FuncDataset()
train_data = data.DataLoader(d_train,
                             batch_size=batch_size,
                             shuffle=True)
total = len(d_train)

optimizer = optim.RMSprop(model.parameters(), lr=0.01)
loss_func = nn.MSELoss()

model.train()
for i in range(epochs):
    for x_train, y_train in train_data:
        y_pred = model(x_train).squeeze()
        error = loss_func(y_pred, y_train)

        optimizer.zero_grad()
        error.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    predictions_all = model(d_train.data).squeeze()
    Q = loss_func(predictions_all, d_train.target).item()
print(Q)
