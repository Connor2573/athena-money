import glob
import pandas as pd

def remove_rows_from_csv(rows=14):
    # Find all CSV files in the data folder
    csv_files = glob.glob('data/*.csv')

    for file in csv_files:
        # Load the CSV file
        df = pd.read_csv(file)

        # Delete the first 14 rows
        df = df.iloc[rows:]

        # Save the modified DataFrame back to the CSV file
        df.to_csv(file, index=False)
        
if __name__ == '__main__':
    remove_rows_from_csv()