from sklearn.metrics import root_mean_squared_error, make_scorer

def rmse_score(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)

rmse_scorer = make_scorer(rmse_score, greater_is_better=False)