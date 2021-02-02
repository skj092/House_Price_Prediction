from sklearn import linear_model
from sklearn import ensemble

models = {
    "linear_regression": linear_model.LinearRegression(),
    "random_forest": ensemble.RandomForestRegressor(n_jobs=-1),
}
