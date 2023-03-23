import os
import pandas as pd

basedir = os.path.abspath(os.path.dirname(__file__))
myWatchList = set(['TSLA', 'EA', 'INTC', 'MSFT', 'SONY', 'NVDA', 'AAPL', 'ENPH', 'GOOGL', 'NOC', 'IBM', 'META', 'CVNA', 
               'AMD', 'AMC', 'AMZN', 'SOFI', 'JPM'])

def loadAllData(path):
    df = None
    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        if os.path.isfile(f) and f.find(".csv") != -1:
            newdf = pd.read_csv(f)
            if df is None:
                df = newdf
            else:
                df = pd.concat([df, newdf])
    return df

def deleteData(path):
    filelist = [ f for f in os.listdir(path) if f.endswith(".csv") ]
    for f in filelist:
        os.remove(os.path.join(path, f))

def singleFileData(watchList):
    #get all data in the myDataMk2 folder
    dataDir = os.path.join(basedir, 'myDataMk2')
    processedDir = os.path.join(basedir, 'processedData')
    df = loadAllData(dataDir)
    df.reset_index(drop=True, inplace=True)
    print(df.dtypes)
    dfs = []
    names = []
    for code in watchList:
        names.append(code)
        dfs.append(df[df['symbol'] == code])

    for name, df in zip(names, dfs):
        df.drop(columns=['symbol'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        for filename in os.listdir(dataDir):
            f = os.path.join(dataDir, filename)
            if os.path.isfile(f) and f.find(name) != -1:
                olddf = pd.read_csv(f)
                df = pd.concat([olddf, df])

        processedPath = os.path.join(processedDir, name + '.csv')
        df.to_csv(processedPath)

    #deleteData(dataDir)


singleFileData(myWatchList)
