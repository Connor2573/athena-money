import pandas as pd
from getRedditData import getRecentHots
import yfinance as yf
from datetime import datetime

directoryToSave = './data/'

myWatchList = ['TSLA']

def buildPandaHourly(codes):
    now = datetime.now()
    now = now.replace(minute = 0, second=0, microsecond=0)
    for code in codes:
        previousFrame = pd.read_csv(directoryToSave + code + '.csv')
        stock = yf.Ticker(code)
        dataframe = pd.DataFrame()
        stock.fast_info
        eForecast = stock.earnings_forecasts
        ## analyst price is very useful indexes: 0: targetLow, 1: current, 2: targetMean, 3: targetHigh, 4: analysts
        analystPrice = stock.analyst_price_target[0]
        earningDates = stock.earnings_dates

        newsList = []
        for news in stock.news:
            newsList.append(news['title'])
        dataframe['timestamp'] = [pd.Timestamp(now)]

        dataframe['priceNow'] = [analystPrice[1]]
        
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
        dataframe = previousFrame.append(dataframe)
        dataframe.to_csv(directoryToSave + code + '.csv', index = False)

def buildPandaFirst(codes):
    now = datetime.now()
    now = now.replace(minute = 0, second=0, microsecond=0)
    for code in codes:
        stock = yf.Ticker(code)
        dataframe = pd.DataFrame()
        stock.fast_info
        eForecast = stock.earnings_forecasts
        ## analyst price is very useful indexes: 0: targetLow, 1: current, 2: targetMean, 3: targetHigh, 4: analysts
        analystPrice = stock.analyst_price_target[0]
        earningDates = stock.earnings_dates

        newsList = []
        for news in stock.news:
            newsList.append(news['title'])
        dataframe['timestamp'] = [pd.Timestamp(now)]

        dataframe['priceNow'] = [analystPrice[1]]
        
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
        dataframe.to_csv(directoryToSave + code + '.csv', index = False)
       
buildPandaFirst(myWatchList)
buildPandaHourly(myWatchList)