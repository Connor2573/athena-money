import getRedditData as rd
import time
import datetime
import pandas as pd
from time import sleep

df = pd.read_csv('./data/stock_market_data/forbes2000/csv/TSLA.csv')
sub = 'investing'

def preprocess(df):
    df['Date'] = pd.to_datetime(df['Date'], dayfirst = True, format = '%d-%m-%Y')

preprocess(df)

for date in df['Date']:
    endDate = date + datetime.timedelta(days=1)
    data = rd.get_posts_for_time_period(sub, date, endDate)
    if not data['data']:
        print('no data')
        sleep
    else:
        