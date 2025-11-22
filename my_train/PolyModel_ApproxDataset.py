import torch
import torch.nn as nn
import torch.utils.data as data


f = lambda t: 0.5 * t + torch.sin(2*t) - 0.1 * torch.exp(t/2) - 12.5 


class ApproxDataset(data.Dataset):
    def __init__(self):
        _x = torch.arange(-6, 6, 0.1, dtype=torch.float32)
        self.data = _x
        self.target = torch.tensor([f(i) for i in self.data],
                                   dtype=torch.float32)
        self.length = len(self.data)

    def __getitem__(self, ind):
        return self.data[ind], self.target[ind]

    def __len__(self):
        return self.length


class PolyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(in_features=3,
                                 out_features=1)

    def forward(self, x):
        x = torch.stack([x, x**2, x**3], dim=1)
        return self.linear1(x).squeeze()


epochs = 20
batch_size = 10
lr = 0.1
model = PolyModel()

d_train = ApproxDataset()
train_data = data.DataLoader(d_train,
                             batch_size=batch_size,
                             shuffle=True
                             )

optimizer = torch.optim.RMSprop(model.parameters(),
                                lr=lr)
loss_func = torch.nn.MSELoss()

model.train()
for e in range(epochs):
    for x_train, y_train in train_data:
        y_pred = model(x_train)
        error = loss_func(y_pred, y_train).squeeze()

        optimizer.zero_grad()
        error.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    predictions = model(d_train.data).squeeze()
    targets = d_train.target
    Q = loss_func(predictions, targets)
print(Q)