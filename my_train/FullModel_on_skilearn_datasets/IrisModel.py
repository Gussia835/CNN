import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn.datasets import load_iris

Iris = load_iris()
_global_var_data_x = torch.tensor(Iris.data, dtype=torch.float32)
_global_var_target = torch.tensor(Iris.target, dtype=torch.long)


class IrisDataset(data.Dataset):
    def __init__(self, data_x=_global_var_data_x,
                 data_target=_global_var_target):
        self.data = data_x
        self.target = data_target
        self.length = len(_global_var_data_x)
        self.categories = ['setosa', 'versicolor', 'virginica']
        self.features = ['sepal length (cm)', 'sepal width (cm)', 
                         'petal length (cm)', 'petal width (cm)']

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return self.length


class IrisModel(nn.Module):
    def __init__(self, in_features=4, out_features=3):
        super().__init__()
        self.linear1 = nn.Linear(in_features=in_features,
                                 out_features=16)
        self.linear2 = nn.Linear(in_features=16,
                                 out_features=out_features)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x


torch.manual_seed(11)

epochs = 10
batch_size = 8
lr = 0.01

d_train = IrisDataset()
train_data = data.DataLoader(d_train,
                             batch_size=batch_size,
                             shuffle=True
                             )

model = IrisModel()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=lr)
loss_func = nn.CrossEntropyLoss()

model.train()
for e in range(epochs):
    for data_x, data_y in train_data:
        y_pred = model(data_x)
        error = loss_func(y_pred, data_y)

        optimizer.zero_grad()
        error.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    all_predictions = model(d_train.data)
    predicted_classes = all_predictions.argmax(dim=1)
    correct = (predicted_classes == d_train.target)
    Q = correct.float().mean().item()
print(Q)
