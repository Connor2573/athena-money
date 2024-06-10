import torch
from chronos import ChronosPipeline
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from utils import read_tickers

def forecast_data(ticker):
    number_to_predict = 30

    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-large",
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )

    # Load the data
    df = pd.read_csv(f'./data/{ticker}.csv', index_col='date')

    df.reset_index(inplace=True, drop=True)

    # context must be either a 1D tensor, a list of 1D tensors,
    # or a left-padded 2D tensor with batch as the first dimension
    context = torch.tensor(df["adjusted_close"][:-number_to_predict])
    prediction_length = number_to_predict
    forecast = pipeline.predict(context, prediction_length, limit_prediction_length=False)  # shape [num_series, num_samples, prediction_length]

    # visualize the forecast
    forecast_index = range(len(df) - number_to_predict, len(df))
    low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

    # Select the last 200 points
    df_last_200 = df["adjusted_close"].tail(200)
    forecast_index_last_200 = forecast_index[-200:] if len(forecast_index) > 200 else forecast_index
    median_last_200 = median[-200:] if len(median) > 200 else median
    low_last_200 = low[-200:] if len(low) > 200 else low
    high_last_200 = high[-200:] if len(high) > 200 else high

    plt.figure(figsize=(10, 6))
    plt.plot(df_last_200.index, df_last_200, color="royalblue", label="historical data")
    plt.plot(forecast_index_last_200, median_last_200, color="tomato", label="median forecast")
    plt.fill_between(forecast_index_last_200, low_last_200, high_last_200, color="tomato", alpha=0.3, label="80% prediction interval")

    # Plot the actual last points
    actual_last = df["adjusted_close"][-number_to_predict:]
    plt.plot(range(len(df) - number_to_predict, len(df)), actual_last, color="green", label=f"actual last {number_to_predict} points")

    plt.legend()
    plt.grid()
    plt.show()

    plt.savefig(f'./figures/chronos_{ticker}_forecast.png')
    
if __name__ == '__main__':
    data_folder = 'data'
    tickers = read_tickers(data_folder)
    
    for ticker in tickers:
        forecast_data(ticker)