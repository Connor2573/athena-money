import quandl

quandl.read_key()
ticker = 'NVDA'
data = quandl.get('WIKI/' + ticker)
data.to_csv('./data/' + ticker + '.csv')