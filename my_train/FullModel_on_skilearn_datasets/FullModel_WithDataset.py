import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim


class MyDataset(data.Dataset):
    def __init__(self,
                 data_x=_global_var_data_x,
                 data_target=_global_var_target):
        self.data = data_x
        self.target = data_target
        self.length = len(data_target)

        self.categories = ['class_0', 'class_1', 'class_2',
                           'class_3', 'class_4', 'class_5',
                           'class_6', 'class_7', 'class_8',
                           'class_9']

    def __getitem__(self, ind):
        return self.data[ind], self.target[ind]

    def __len__(self):
        return self.length


class MyModel(nn.Module):
    def __init__(self, in_features, hidden_layer1, hidden_layer2, out_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features=in_features,
                                 out_features=hidden_layer1)
        self.linear2 = nn.Linear(in_features=hidden_layer1,
                                 out_features=hidden_layer2)
        self.linear3 = nn.Linear(in_features=hidden_layer2,
                                 out_features=out_features)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x


epochs = 10
lr = 0.01
batch_size = 12

d_train = MyDataset()
train_data = data.DataLoader(d_train,
                             batch_size=batch_size,
                             shuffle=True)

loss_func = nn.CrossEntropyLoss()
model = MyModel(8*8, 32, 16, 10)
optimizer = optim.Adam(model.parameters(),
                       lr=lr)

model.train()
for _e in range(epochs):
    for x_data, y_data in train_data:
        predictions = model(x_data)
        error = loss_func(predictions, y_data)

        optimizer.zero_grad()
        error.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    all_data = d_train.data
    predictions = model(all_data)
    corrects = (predictions == d_train.target).float()
    Q = corrects.mean().item()
print(Q)
