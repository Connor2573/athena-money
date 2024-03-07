import os
import matplotlib.pyplot as plt
import csv
from datetime import datetime

# Create the "figures" folder if it doesn't exist
if not os.path.exists("figures"):
    os.makedirs("figures")

# Get the list of files in the "data" folder
data_files = os.listdir("data")

# Create a new figure
plt.figure()

# Iterate over each file in the "data" folder
for file_name in data_files:
    # Read the data from the file
    data = []
    with open(os.path.join("data", file_name), "r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the first line
        for row in reader:
            date = datetime.strptime(row[0], "%Y-%m-%d")  # Convert the date string to a datetime object
            close_value = float(row[4])  # Convert the "Close" value to a float
            data.append([date, close_value])

    # Visualize the data
    dates = [row[0] for row in data]
    close_values = [row[1] for row in data]
    plt.plot(dates, close_values, label=file_name)

# Set the title, labels, and legend
plt.title("Stock Data")
plt.xlabel("Date")
plt.ylabel("Close Value")
plt.legend()

# Autoscale the X and Y axis
plt.autoscale()

# Save the plot into the "figures" folder
plt.savefig(os.path.join("figures", "stocks.png"))

# Close all open plots
plt.close("all")