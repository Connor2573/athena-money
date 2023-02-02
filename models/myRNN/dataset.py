import torch
import pandas as pd
import os


def loadAllCsvs(path):
    dfs = []
    for filename in os.listdir(path):
        f = os.path.join(path, file)
        if os.path.isfile(f) and f.find(".csv") != -1:
            dfs.append(pd.read_csv(f))
    return dfs

def preprocess(df):
    df['Date'] = pd.to_datetime(df['Date'], dayfirst = True, format = '%d-%m-%Y')

class teslaData(torch.utils.data.Dataset):
    def __init(self):
        df = pd.read_csv('./data/stock_market_data/forbes2000/csv/TSLA.csv')
        preprocess(df)

        self.close = df['Close']
        self.open = df['Open']

    def __len__(self):
        return len(self.open)

    def __getitem__(self, index, device='cpu'):
        return {
            'Open': torch.tensor(self.open[index], device=device),
            'Close': torch.tensor(self.close[index], device=device)
            }




#NOT DONE
class forbesData(torch.utils.data.Dataset):
    def __init__(self):
        path = './data/stock_market_data/forbes2000/csv/'
