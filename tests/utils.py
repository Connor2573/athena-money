import os

def read_tickers(data_folder):
    # List all files in the data folder
    files = os.listdir(data_folder)
    
    # Extract ticker names from file names
    tickers = [file.split('.')[0] for file in files if file.endswith('.csv')]
    
    return tickers