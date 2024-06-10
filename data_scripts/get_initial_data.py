import requests
import pandas as pd

#read the api key from the config file
import configparser

config = configparser.ConfigParser()
config.read('api_keys.cfg')

alphavantage_api_key = config['API_KEYS']['alphavantage']

ticker = 'IBM'
outputsize = 'full'

# actually do stuff
# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&apikey={alphavantage_api_key}&outputsize={outputsize}'
r = requests.get(url)
data = r.json()

# Extract the time series data
time_series_data = data['Time Series (Daily)']

# Convert the dictionary to a DataFrame
df = pd.DataFrame.from_dict(time_series_data).T

# Reset the index
df = df.reset_index()

# Now the dates are in a column named 'index'. You can rename this column to 'date':
df = df.rename(columns={'index': 'date'})

# Sort by 'date' in ascending order
df = df.sort_values('date')

# Rename the other columns
df = df.rename(columns={
    '1. open': 'open',
    '2. high': 'high',
    '3. low': 'low',
    '4. close': 'close',
    '5. adjusted close': 'adjusted_close',
    '6. volume': 'volume',
    '7. dividend amount': 'dividend_amount',
    '8. split coefficient': 'split_coefficient'
})

# Save the DataFrame to a CSV file
df.to_csv(f'data/{ticker}.csv', index=False)