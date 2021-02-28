import torch
import torch.nn as nn
import torch.nn.functional as F


a = -0.2
b = -0.3


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout):

        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.linears = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.dropout = dropout

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linears.append(nn.Linear(input_dim, output_dim))
        else:
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        self.input_norm = nn.BatchNorm1d(input_dim)
        # self.reset_param()

    def reset_param(self):
        for l in self.linears:
            l.weight.data.uniform_(a, b)
            l.bias.data.uniform_(a, b)

    def forward(self, x):
        h = x
        # h = F.dropout(h, .5, self.training)
        # h = self.input_norm(h)
        for layer in range(self.num_layers - 1):
            h = self.linears[layer](h)
            h = self.batch_norms[layer](h)
            h = F.relu(h)
            # h = F.tanh(h)
            h = F.dropout(h, self.dropout, self.training)
        return self.linears[self.num_layers - 1](h)
