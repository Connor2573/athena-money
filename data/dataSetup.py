import pandas as pd
import matplotlib.pyplot as plt

pd.option_context('display.max_columns', None, 'display.max_rows', None)
df = pd.read_csv('./data/stock_market_data/forbes2000/csv/TSLA.csv')

def preprocess(df):
    df['Date'] = pd.to_datetime(df['Date'], dayfirst = True, format = '%d-%m-%Y')


preprocess(df)
print(df.head())
print(df.dtypes)
print(df.describe())

print(df['Close'])
ts = pd.Series(df['Close'].values, index=df['Date'])
ts.plot()
plt.show()