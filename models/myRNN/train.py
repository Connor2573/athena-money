import torch
import os
import pandas as pd
from models import LSTM1, MV_LSTM
from torch.utils.data import DataLoader
from myStockDataset import myData, loadAllMyCsvsIntoDataframeList

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dfs = loadAllMyCsvsIntoDataframeList('./data/processedData')
loaders = []
for df in dfs:
    dataset = myData(df)
    loader = DataLoader(dataset)
    loaders.append(loader)

features = 21
seq_len = 10

net = MV_LSTM(features, seq_len)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-1)

net.to(device)
net.train()
epochs = 100
for epoch in range(epochs):
    for loader in loaders:
        for data in loader:
            inpt = data['input'].to(device)
            target = data['output'].to(device)
            inpt = inpt.unsqueeze(1)
            batchSize = inpt.size(0)
            #print(batchSize)
            net.init_hidden(batchSize)

            output = net(inpt)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    print('Epoch: ', epoch, 'loss: ', loss.item())

torch.save(net, './mk1/')