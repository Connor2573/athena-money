import pandas as pd
import os
import matplotlib.pyplot as plt

path = './data/myData/'

def preprocess(df):
    df['Date'] = pd.to_datetime(df['Date'], dayfirst = True, format = '%d-%m-%Y')

dfs = []
for filename in os.listdir(path):
    f = os.path.join(path, filename)
    if os.path.isfile(f) and f.find(".csv") != -1:
        dfs.append(pd.read_csv(f))

#this kind of works but ehh
ax = dfs[0].plot(x='timestamp', y='priceNow')
for df in dfs:
    ax = df.plot(x='timestamp', y='priceNow', ax=ax)
plt.show()
        