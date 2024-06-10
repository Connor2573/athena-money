import matplotlib.pyplot as plt
import timesfm
import os
import pandas as pd
from utils import read_tickers

def forecast_data(ticker):
    input_length=8192
    prediction_length=32
    tfm = timesfm.TimesFm(
        context_len=input_length,
        horizon_len=prediction_length,
        input_patch_len=32,
        output_patch_len=128,
        num_layers=20,
        model_dims=1280,
        backend="gpu",
    )
    tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")

    raw_df = pd.read_csv(f"data/{ticker}.csv")

    raw_df = raw_df.rename(columns={"date": "ds"})
    # Convert 'ds' column to datetime
    raw_df['ds'] = pd.to_datetime(raw_df['ds'])
    # Add a 'unique_id' column to input_df
    raw_df['unique_id'] = 'D0'

    # Select the range from raw_df
    input_df = raw_df.tail(input_length + prediction_length)

    # Data for forecasting
    forecast_input_df = input_df.iloc[:-prediction_length]

    forecast_df = tfm.forecast_on_df(inputs=forecast_input_df, freq='D', value_name="adjusted_close", num_jobs=1)

    # Create a directory for the figures if it doesn't exist
    if not os.path.exists('figures'):
        os.makedirs('figures')
        
    # Number of rows to plot
    num_rows = 200

    # Plotting the input and forecast on the same graph
    plt.figure(figsize=(10, 5))
    plt.plot(input_df.tail(num_rows)['ds'], input_df.tail(num_rows)['adjusted_close'], label='Input', color='blue')
    plt.plot(forecast_df.tail(num_rows)['ds'], forecast_df.tail(num_rows)['timesfm'], label='Forecast', color='red')

    # Plotting the quantile columns
    quantile_columns = ['timesfm-q-0.1', 'timesfm-q-0.9']
    for column in quantile_columns:
        plt.plot(forecast_df.tail(num_rows)['ds'], forecast_df.tail(num_rows)[column], label=column, alpha=0.5)

    plt.legend()
    # Save the figure to the figures directory
    plt.savefig(f'figures/google_{ticker}_forecast.png')
    plt.close()

if __name__ == '__main__':
    data_folder = 'data'
    tickers = read_tickers(data_folder)
    
    for ticker in tickers:
        forecast_data(ticker)