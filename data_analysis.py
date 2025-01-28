import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("data/cars.csv")

numeric_columns = data.select_dtypes(include=["float64", "int64"]).columns
categorical_columns = data.select_dtypes(include=["object", "bool"]).columns


for col in numeric_columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(data[col], kde=True, bins=30, color='blue')
    plt.title(f"Rozkład: {col}")
    plt.xlabel(col)
    plt.ylabel("Liczba obserwacji")
    plt.show()

for col in categorical_columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=data, x=col, color='blue')
    plt.title(f"Częstotliwość występowania: {col}")
    plt.xlabel(col)
    plt.ylabel("Liczba obserwacji")
    plt.xticks([])
    plt.show()
