import pandas as pd
import os

pd.set_option('display.max_columns', None)

path = './data/myData/'

def loadSpecificCsv(csv):
    return [pd.read_csv(path + csv)], [csv]

def loadAllData():
    dfs = []
    names = []
    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        if os.path.isfile(f) and f.find(".csv") != -1:
            dfs.append(pd.read_csv(f))
            names.append(filename)
    return dfs, names

def processTime(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'], yearfirst=True, infer_datetime_format=True)
    df['lastEarningDate'] = pd.to_datetime(df['lastEarningDate'], yearfirst=True, infer_datetime_format=True)
    df['nextEarningDate'] = pd.to_datetime(df['nextEarningDate'], yearfirst=True, infer_datetime_format=True)

def processText(df):
    from robertaSentimentModel import getSentiment
    posAvg = []
    negAvg = []
    neuAvg = []
    pInd = 0
    negInd = 1
    neuInd = 2
    for titles in df['text']:
        avgs = [0, 0, 0]

        count = 0
        for title in titles:
            sentiment = getSentiment(title)
            avgs[pInd] += sentiment['positive']
            avgs[negInd] += sentiment['negative']
            avgs[neuInd] += sentiment['neutral']
            count = count + 1
        posAvg.append(avgs[pInd] / count)
        negAvg.append(avgs[negInd] / count)
        neuAvg.append(avgs[neuInd] / count)
    return posAvg, negAvg, neuAvg

def preprocess(df):
    processTime(df)
    df['positiveSentiment'], df['negativeSentiment'], df['neutralSentiment'] = processText(df)


#dfs, names = loadAllData()
dfs, names = loadSpecificCsv('TSLA.csv')
for df, name in zip(dfs, names):
    processTime(df)
    print(df.head())
    print(df.dtypes)
