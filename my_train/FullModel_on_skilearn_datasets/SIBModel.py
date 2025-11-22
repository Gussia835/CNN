# Для прогноза сахара в крови (Sugar in blood = SIB)
import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn.datasets import load_diabetes


class SIB_Dataset(data.Dataset):

    def __init__(self, data_x, target):
        self.data = torch.tensor(data_x, dtype=torch.float32)
        self.target = torch.tensor(target, dtype=torch.float32)
        self.length = len(self.data)

    def __getitem__(self, ind):
        return self.data[ind], self.target[ind]

    def __len__(self):
        return self.length


class SIB_RegressionModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=10,
                                out_features=64)
        self.layer2 = nn.Linear(in_features=64,
                                out_features=1)

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = self.layer2(x)
        return x


diabetes = load_diabetes()

_global_var_data_x = torch.tensor(diabetes.data, dtype=torch.float32)
_global_var_target = torch.tensor(diabetes.target, dtype=torch.float32)


batch_size = 8
epochs = 10

model = SIB_RegressionModel()
d_train = SIB_Dataset(_global_var_data_x, _global_var_target)
train_data = data.DataLoader(d_train,
                             batch_size=batch_size,
                             shuffle=True,
                             )

optimizer = torch.optim.RMSprop(model.parameters(),
                                lr=0.01)
loss_func = nn.MSELoss()

model.train()
for e in range(epochs):
    for x_train, y_train in train_data:
        y_pred = model(x_train).squeeze()
        error = loss_func(y_pred, y_train)

        optimizer.zero_grad()
        error.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    all_data = torch.tensor(_global_var_data_x, dtype=torch.float32)
    all_targets = torch.tensor(_global_var_target, dtype=torch.float32)
    predictions = model(all_data).squeeze()
    Q = loss_func(predictions, all_targets).item()

print(Q)
