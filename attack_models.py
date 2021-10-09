from torch import nn


class MLP(nn.Module):
    def __init__(self, in_size, out_size, hidden_1, hidden_2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_size, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.out = nn.Linear(hidden_2, out_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.batchnorm1 = nn.BatchNorm1d(hidden_2)
        self.batchnorm2 = nn.BatchNorm1d(hidden_2)

    def forward(self, inputs):
        x = self.relu(self.fc1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.fc2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.out(x)
        return x