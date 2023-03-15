import pandas as pd
import os
from robertaSentimentModel import getSentiment

pd.set_option('display.max_columns', None)

path = './data/myData/'
pathToProcessed = './data/processedData/'

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
    timeCols = ['timestamp']#, 'lastEarningDate', 'nextEarningDate'] we are not using these right now
    for tCol in timeCols:
        df[tCol] = pd.to_datetime(df[tCol], yearfirst=True, infer_datetime_format=True)
        

def processText(df):
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

def preprocess(df, name):
    processTime(df)
    pos, neg, neu = processText(df)
    df['positiveScore'] = pos
    df['negativeScore'] = neg
    df['neutralScore'] = neu

    #gonna drop some data we are not ready to use
    columnsToRemove = ['lastEarningDate', 'lastEarnExp', 'lastEarnActual', 'lastEarnSuprise%', 'nextEarningDate', 'nextEarnExepected', 'text']
    df.drop(columns=columnsToRemove, inplace=True)

    df.to_csv(pathToProcessed + name, index=False)




dfs, names = loadAllData()
#dfs, names = loadSpecificCsv('TSLA.csv')
for df, name in zip(dfs, names):
    print('working on: ' + name)
    preprocess(df, name)
    print('wrote: ' + name)

print('done')
