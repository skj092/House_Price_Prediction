# ohe_lr.py
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing
import joblib
import os
import config


def run(fold):
    df = pd.read_csv("../input/train_fold.csv")
    df.SalePrice = np.log(df.SalePrice)

    features = [f for f in df.columns if f not in ("id", "SalePrice", "kfold")]

    # replacing missing values with "NONE"
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    ohe = preprocessing.OneHotEncoder()
    full_data = pd.concat([df_train[features], df_valid[features]], axis=0)

    ohe.fit(full_data[features])

    x_train = ohe.transform(df_train[features])
    x_valid = ohe.transform(df_valid[features])

    model = linear_model.LinearRegression()
    model.fit(x_train, df_train.SalePrice.values)

    valid_preds = model.predict(x_valid)
    mse = metrics.mean_squared_error(df_valid.SalePrice.values, valid_preds)

    print(f"Fold = {fold}, AUC = {mse}")

    # Saving the model
    # saving the model
    joblib.dump(model, os.path.join(config.MODEL_OUTPUT, f"rf.bin"))


if __name__ == "__main__":
    for fold in range(5):
        run(fold)
