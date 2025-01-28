# Source https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes?select=cclass.csv

import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import root_mean_squared_error, make_scorer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from helpers import rmse_scorer
import joblib


def training_base(X_train, X_test, y_train, y_test, suffix):
    random_state = 1

    num_pipeline = Pipeline(
        [
            ("num_imputer", SimpleImputer(strategy="mean")),
            ("std_scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        [
            ("cat_imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot_encoder",
                OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
            ),
        ]
    )

    num_attribute = X_train.select_dtypes(include=["float64", "int64"]).columns
    cat_attribute = X_train.select_dtypes(include=["object", "bool"]).columns

    full_pipeline = ColumnTransformer(
        [("num", num_pipeline, num_attribute), ("cat", cat_pipeline, cat_attribute)]
    )

    X_train_prepared = full_pipeline.fit_transform(X_train)
    X_test_prepared = full_pipeline.transform(X_test)


    models = {
        "RandomForest": {
            "model": RandomForestRegressor(random_state=random_state),
            "params": {"max_depth": [10, 20, None,30,40]},
        },
        "GradientBoosting": {
            "model": GradientBoostingRegressor(random_state=random_state),
            "params": {"learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 5, 10, 20]},
        },
        "MLPRegressor": {
            "model": MLPRegressor(random_state=random_state, max_iter=1000),
            "params": {
                "hidden_layer_sizes": [(50,), (50, 50), (50, 50, 50)],
                "alpha": [0.0001, 0.001, 0.01, 0.1],
            },
        },
        "LinearRegression": {"model": LinearRegression(), "params": {}},
    }

    best_models = {}
    for name, model_info in models.items():
        print(f"Training {name}...")
        grid_search = GridSearchCV(
            model_info["model"], model_info["params"], cv=5, scoring=rmse_scorer,n_jobs=-1
        )
        start_time = time.time()
        grid_search.fit(X_train_prepared, y_train)
        end_time = time.time()
        training_time = end_time - start_time
        best_models[name] = grid_search
        cv_results = grid_search.cv_results_

        results_df = pd.DataFrame(
            {
                "param_combination": [str(params) for params in cv_results["params"]],
                "mean_test_score": cv_results["mean_test_score"],
                "std_test_score": cv_results["std_test_score"],
                "mean_fit_time": cv_results["mean_fit_time"],
                "mean_score_time": cv_results["mean_score_time"],
            }
        )

        results_df = results_df.sort_values(by="mean_test_score", ascending=False)
        results_df.to_csv(f"stats/{name}_stats.csv", index=False)
        print(f"Najlepszy wynik CV (RMSE) dla {name}: {grid_search.best_score_:.4f}")
        print(f"Najlepsze parametry dla {name}: {grid_search.best_params_}")
        print(f"Czas trenowania {name}: {training_time:.2f} s")

    for name, grid_search in best_models.items():
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test_prepared)
        test_rmse = root_mean_squared_error(y_test, y_pred)
        print(f"{name} => Test RMSE: {test_rmse:.2f}")

    joblib.dump(best_models, f"models/trained_models_{suffix}.pkl")
    joblib.dump(full_pipeline, f"models/full_pipeline_{suffix}.pkl")
