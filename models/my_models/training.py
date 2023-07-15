import torch
import os
import pandas as pd
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

prices = (pd.read_hdf('./data/assets.h5', 'quandl/wiki/prices').adj_close.unstack().loc['2000':])
prices.info()
returns = (prices
           .resample('M')
           .last()
           .pct_change()
           .dropna(how='all')
           .loc['2000': '2017']
           .dropna(axis=1)
           .sort_index(ascending=False))
returns = returns.where(returns<1).dropna(axis=1)
returns.info()