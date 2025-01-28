import joblib
import pandas as pd

from sklearn.metrics import root_mean_squared_error, make_scorer

def rmse_score(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)

rmse_scorer = make_scorer(rmse_score, greater_is_better=False)

best_models = joblib.load("models/trained_models.pkl")
full_pipeline = joblib.load("models/full_pipeline.pkl")

data = [["Q7", 2020, "Semi-Auto", 10, "Diesel",  33.2, 3.0]]

columns = ["model", "year", "transmission", "mileage", "fuelType", "mpg", "engineSize"]

data_df = pd.DataFrame(data, columns=columns)


data_prepared = full_pipeline.transform(data_df)

output_ml = best_models["MLPRegressor"].predict(data_prepared)
output_rf = best_models["RandomForest"].predict(data_prepared)
output_lr = best_models["LinearRegression"].predict(data_prepared)
output_gb = best_models["GradientBoosting"].predict(data_prepared)

print(f"Przewidywana cena ml: {output_ml[0]:.2f}")
print(f"Przewidywana cena rf: {output_rf[0]:.2f}")
print(f"Przewidywana cena lr: {output_lr[0]:.2f}")
print(f"Przewidywana cena gb: {output_gb[0]:.2f}")