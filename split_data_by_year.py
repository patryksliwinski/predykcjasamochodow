import pandas as pd

data = pd.read_csv("data/cars.csv")

later_2018 = data[data["year"] >= 2019]

older_2018 = data[data["year"] < 2019]


older_2018.to_csv("data/cars_older_2018.csv", index=False)
later_2018.to_csv("data/cars_later_2018.csv", index=False)
