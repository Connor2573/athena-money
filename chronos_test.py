import torch
from chronos import ChronosPipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ticker = 'NVDA'
number_to_predict = 64

pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-large",
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)

# Load the data
df = pd.read_csv(f'./data/{ticker}.csv')

# context must be either a 1D tensor, a list of 1D tensors,
# or a left-padded 2D tensor with batch as the first dimension
context = torch.tensor(df["5. adjusted close"][:-number_to_predict])
prediction_length = number_to_predict
forecast = pipeline.predict(context, prediction_length)  # shape [num_series, num_samples, prediction_length]

# visualize the forecast
forecast_index = range(len(df) - number_to_predict, len(df))
low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

plt.figure(figsize=(8, 4))
plt.plot(df["5. adjusted close"], color="royalblue", label="historical data")
plt.plot(forecast_index, median, color="tomato", label="median forecast")
plt.fill_between(forecast_index, low, high, color="tomato", alpha=0.3, label="80% prediction interval")

# Plot the actual last 10 points
actual_last_10 = df["5. adjusted close"][-number_to_predict:]
plt.plot(range(len(df) - number_to_predict, len(df)), actual_last_10, color="green", label="actual last 10 points")

plt.legend()
plt.grid()
plt.show()

plt.savefig(f'./figures/{ticker}_forecast.png')