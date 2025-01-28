import os
import pandas as pd

csv_directory = "cars"

dataframes = []

for filename in os.listdir(csv_directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(csv_directory, filename)
        df = pd.read_csv(file_path)
        dataframes.append(df)

combined_df = pd.concat(dataframes, ignore_index=True)

output_path = "data/cars.csv"
combined_df.to_csv(output_path, index=False)