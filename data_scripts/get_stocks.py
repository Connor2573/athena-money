import yfinance as yf
import os
import ta

# Define the list of companies
companies = ['TSLA', 'NVDA', 'META', 'NOC', 'AMZN', 'MSFT', 'GOOGL', 'AAPL', 'TSM', 'BABA', 'T', 'VZ', 'TMUS', 'INTC', 'CSCO', 'ORCL', 'IBM', 'QCOM', 'ADBE', 'CRM', 'NVDA', 'PYPL', 'ACN', 'TXN', 'AVGO', 'INTU', 'MU', 'LRCX']

# Create the data folder if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

# Fetch stock data for each company and save as CSV
for company in companies:
    data = yf.download(company, start='2022-01-01', end='2024-01-01')
    
    # Calculate SMA, EMA, MACD, and RSI
    data['SMA'] = ta.trend.sma_indicator(data['Close'], window=14)
    data['EMA'] = ta.trend.ema_indicator(data['Close'], window=14)
    data['MACD'] = ta.trend.MACD(data['Close']).macd()
    data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
    
    data.to_csv(f'data/{company}.csv')