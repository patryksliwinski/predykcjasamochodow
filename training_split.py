# Source https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes

import pandas as pd
from training_base import training_base

data = pd.read_csv("data/cars_older_2018.csv")
test = pd.read_csv("data/cars_later_2018.csv")

X_train = data.drop(columns=["price", "tax"])
y_train = data["price"]
X_test = test.drop(columns=["price","tax"])
y_test = test["price"]


training_base(X_train,X_test,y_train,y_test,suffix="split")