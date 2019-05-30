# Imports from external libraries

# Imports from internal libraries


import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class LSTMs(nn.Module):
    def __init__(self, emb_dim, hidden_dim, num_layers, batch_size, device):
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.LSTM1 = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.LSTM2 = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(2 * self.hidden_dim, 2)

        self.hidden1 = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(device), \
                       torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(device)
        self.hidden2 = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(device), \
                       torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(device)

    def forward(self, x):
        # lstm takes batch_size x sequence_lenght x embedding_size
        # h and c are  num_layers x batch_size x hidden_size
        # lstm outputs batch_size x sequence_lenght x embedding_size
        out1, self.hidden1 = self.LSTM1(x[0], self.hidden1)
        out2, self.hidden2 = self.LSTM2(x[1], self.hidden2)

        cell_merged = torch.cat((self.hidden1[1], self.hidden2[1]), dim=2)
        output = self.fc(cell_merged)
        return output


if __name__ == '__main__':
    batch_size = 3
    embed_size = 5
    hidden_size = 4
    num_layers = 1

    lstms = LSTMs(embed_size, hidden_size, num_layers, batch_size)
    input1, input2 = torch.rand(batch_size, 4, embed_size), torch.rand(batch_size, 6, embed_size),
    lstms.forward((input1, input2))


