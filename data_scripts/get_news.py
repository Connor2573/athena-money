import requests
import os
from datetime import timedelta, datetime
import pandas as pd
from tqdm import tqdm
import time
import random
import glob

def get_top_stories(date, query='stocks', num_articles=3):
    
    url = 'https://api.polygon.io/v2/reference/news'
    params = {
        'ticker': query,
        'limit': num_articles,
        'apiKey': os.environ['POLYGON_IO_KEY'],
        'published_utc': date.strftime('%Y-%m-%d'),
    }

    response = requests.get(url, params=params)
    data = response.json()
    return data['results']

# Find all CSV files in the data folder
csv_files = ['data/NVDA.csv', 'data/TSLA.csv', 'data/MSFT.csv', 'data/AAPL.csv', 'data/AMZN.csv', 'data/GOOGL.csv']
#csv_files = glob.glob('./data/*.csv')
pbar = tqdm(csv_files)

for file in pbar:
    ticker = file.split('/')[1].split('.')[0]
    # Load the CSV file
    df = pd.read_csv(file)

    # Extract the dates
    dates = df['Date'].tolist()

    # Convert strings to datetime objects
    dates = [datetime.strptime(date, '%Y-%m-%d') for date in dates]

    # Initialize an empty DataFrame to store the news data
    news_df = pd.DataFrame()
    
    for date in dates:
        time.sleep(12 + random.randint(0, 9))
        data = get_top_stories(date, ticker, 3)
        pbar.set_description(f"Ticker: {ticker}, Date: {date.strftime('%Y-%m-%d')}, ")
        # Convert the JSON data to a DataFrame and append it to news_df
        json_data_df = pd.json_normalize(data)
        news_df = pd.concat([news_df, json_data_df], ignore_index=True)

    # Save the news data to a CSV file
    news_df.to_csv(f'./data/news/{ticker}.csv', index=False)
    