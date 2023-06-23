import quandl

quandl.read_key()
data = quandl.get('WIKI/FB')
data.to_csv('./data/FB.csv')