import torch
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder


def preprocess(df):
    preprocesstimestamp(df)
    df = df.drop(columns=['lastClose'])

def loadAllMyCsvsIntoSingleDataframe(path):
    dfs = []
    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        if os.path.isfile(f) and f.find(".csv") != -1:
            name = filename.replace('.csv', '')
            df = pd.read_csv(f)
            df['name'] = name
            dfs.append(df)

    
    encoder = LabelEncoder()
    df = pd.DataFrame()
    for ndf in dfs:
        df = pd.concat([df, ndf])
    df['name'] = encoder.fit_transform(df['name'])
    return df

def loadSingleCsv(filename):
    path = './data/processedData/'
    f = os.path.join(path, filename)
    return pd.read_csv(f)

def loadAllMyCsvsIntoDataframeList(path):
    dfs = []
    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        if os.path.isfile(f) and f.find(".csv") != -1:
            df = pd.read_csv(f)
            dfs.append(df)
    return dfs

def preprocesstimestamp(df):
    time_id = []
    count = 0
    for timestamp in df['timestamp']:
        time_id.append(count)
        count = count + 1
    df.drop(inplace=True , columns=['timestamp'])
    df['timestamp'] = time_id


class myData(torch.utils.data.Dataset):
    def __init__(self, df):
        preprocess(df)
        self.input = df.drop(columns='price').values
        self.output = df['price'].values
        self.dataframe = df
        self.numFeatures = len(self.input)
        self.numTargets = len(self.output)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        input = torch.tensor(self.input[index], dtype=torch.float32)
        output = torch.tensor(self.output[index], dtype=torch.float32)
        return {'input':input, 'output': output}

     
