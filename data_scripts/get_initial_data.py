import requests
import pandas as pd

#read the api key from the config file
import configparser

config = configparser.ConfigParser()
config.read('api_keys.cfg')

alphavantage_api_key = config['API_KEYS']['alphavantage']

ticker = 'AMD'
# actually do stuff
# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&apikey={alphavantage_api_key}&outputsize=full'
r = requests.get(url)
data = r.json()

# Extract the time series data
time_series_data = data['Time Series (Daily)']

# Convert the dictionary to a DataFrame
df = pd.DataFrame.from_dict(time_series_data).T

# Save the DataFrame to a CSV file
df.to_csv(f'data/{ticker}.csv')