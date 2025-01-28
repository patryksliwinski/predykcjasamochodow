# Source https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes

import pandas as pd
from sklearn.model_selection import train_test_split
from training_base import training_base

random_state = 1

data = pd.read_csv("data/cars.csv")

X = data.drop(columns=["price", "tax"])
y = data["price"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state
)
training_base(X_train,X_test,y_train,y_test,suffix="random")