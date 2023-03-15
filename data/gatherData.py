import pandas as pd
import yfinance as yf
from datetime import datetime
import os

directoryToSave = '/home/xarga/repos/athena-money/data/myData/'
#directoryToSave = './data/myData/'

myWatchList = set(['TSLA', 'EA', 'INTC', 'MSFT', 'SONY', 'NVDA', 'AAPL', 'ENPH', 'GOOGL', 'NOC', 'IBM', 'META', 'CVNA', 
               'AMD', 'AMC', 'AMZN', 'SOFI', 'JPM'])

def buildPandaHourly(codes):
    now = datetime.now()
    now = now.replace(microsecond = 0)
    for code in codes:
        found = os.path.exists(directoryToSave + code + '.csv')
        if found:
            previousFrame = pd.read_csv(directoryToSave + code + '.csv')
        stock = yf.Ticker(code)
        try:
            stock.info
        except:
            print("Not valid stock: {}", code)
            continue

        dataframe = pd.DataFrame()
        eForecast = stock.earnings_forecasts
        ## analyst price is very useful indexes: 0: targetLow, 1: current, 2: targetMean, 3: targetHigh, 4: analysts
        analystPrice = stock.analyst_price_target[0]
        earningDates = stock.earnings_dates

        newsList = []
        for news in stock.news:
            newsList.append(news['title'])
        dataframe['timestamp'] = [pd.Timestamp(now)]

        info = stock.fast_info
        dataframe['price'] = info['last_price']
        dataframe['open'] = info['open']
        dataframe['lastClose'] = info['previous_close']
        dataframe['dayHigh'] = info['day_high']
        dataframe['dayLow'] = info['day_low']
        dataframe['yearLow'] = info['year_low']
        dataframe['yearHigh'] = info['year_high']
        dataframe['50avg'] = info['fifty_day_average']
        dataframe['200avg'] = info['two_hundred_day_average']
        
        dataframe['targetLow'] = [analystPrice[0]]
        dataframe['targetMean'] = [analystPrice[2]]
        dataframe['targetHigh'] = [analystPrice[3]]

        dataframe['forecastNowAvg'] = eForecast.avg[0]
        dataframe['forecastNowGrowth'] = eForecast.growth[0]
        dataframe['forecastFutureAvg'] = eForecast.avg[1]
        dataframe['forecastFutureGrowth'] = eForecast.avg[1]
        dataframe['forecastFutureYrAvg'] = eForecast.avg[2]
        dataframe['forecastFutureYrGrowth'] = eForecast.avg[2]

        nextEarnings = earningDates.values[4]
        lastEarnings = earningDates.values[5]
        
        dataframe['lastEarningDate'] = earningDates.index[5]
        dataframe['lastEarnExp'] = lastEarnings[0]
        dataframe['lastEarnActual'] = lastEarnings[1]
        dataframe['lastEarnSuprise%'] = lastEarnings[2]
        
        dataframe['nextEarningDate'] = earningDates.index[4]
        dataframe['nextEarnExepected'] = nextEarnings[0]

        dataframe['text'] = [newsList]
        if found:
            dataframe = pd.concat([previousFrame, dataframe])
        dataframe.to_csv(directoryToSave + code + '.csv', index = False)

buildPandaHourly(myWatchList)
