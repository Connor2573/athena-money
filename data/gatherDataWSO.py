import requests
import pandas as pd
from datetime import datetime

directoryToSave = '/home/xarga/repos/athena-money/data/myDataMk2/'
#tmpDirToSave = "D:\\Programming\\AthenaMoney\\data\\myDataMk2\\"

now = datetime.now()
myWatchList = set(['TSLA', 'EA', 'INTC', 'MSFT', 'SONY', 'NVDA', 'AAPL', 'ENPH', 'GOOGL', 'NOC', 'IBM', 'META', 'CVNA', 
               'AMD', 'AMC', 'AMZN', 'SOFI', 'JPM'])

fields = set(['symbol', 'price', 'percentChange', 'open', 'high', 'low', 'relVolume'])


urlMain = 'https://www.wallstreetoddsapi.com/api/livestockprices?&apikey=umen2wfalwyr'
urlfields = '&fields='
format = '&format=json'
symbols = '&symbols='

for field in fields:
    urlfields += field + ','
urlfields = urlfields[:-1]

for symbol in myWatchList:
    symbols += symbol + ','
symbols = symbols[:-1]

url = urlMain+urlfields+format+symbols
r = requests.post(url)
json = r.json()

df = pd.DataFrame(json['response'])

newFields = '&fields=symbol,watchers,hourChange,sentiment'
stockTwitUrl = 'https://www.wallstreetoddsapi.com/api/stocktwits?apikey=umen2wfalwyr' + newFields + format + symbols

r = requests.post(stockTwitUrl)
json = r.json()

new_df = pd.DataFrame(json['response'])

merged = pd.merge(df, new_df, on = 'symbol')

merged.to_csv(directoryToSave + now.strftime('%Y%m%d%H%M') + '.csv', index = False)
