# ohe_lr.py
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing
import joblib
import os
import config
import pickle


def run(fold):
    df = pd.read_csv("../input/train_fold.csv")
    df.SalePrice = np.sqrt(df.SalePrice)
    df_test = pd.read_csv('../input/test.csv')

    features = [f for f in df.columns if f not in ("id", "SalePrice", "kfold")]

    # replacing missing values with "NONE"
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")
        df_test.loc[:, col] = df_test[col].astype(str).fillna("NONE")
        
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # initializing one hot encoder
    ohe = preprocessing.OneHotEncoder()

    # concatenating whole data
    full_data = pd.concat([df_train[features], df_valid[features], df_test[features]], axis=0)

    # fitting one hot encoder
    ohe.fit(full_data[features])

    # transforming all the columns into one hot encoded features
    x_train = ohe.transform(df_train[features])
    x_valid = ohe.transform(df_valid[features])
    df_test = ohe.transform(df_test[features])
    
    # saving preprocessed test data for prediction
    pickle.dump(df_test, open('../input/test_processed.pkl', 'wb'))

    # initializing linear regression
    model = linear_model.LinearRegression()

    # fitting model to training dataset
    model.fit(x_train, df_train.SalePrice.values)

    # prediction on validation dataset
    valid_preds = model.predict(x_valid)

    # mean square error
    mse = metrics.mean_squared_error(df_valid.SalePrice.values, valid_preds)

    # output
    print(f"Fold = {fold}, AUC = {mse}")

    # Saving the model
    joblib.dump(model, os.path.join(config.MODEL_OUTPUT, f"rf.bin"))


if __name__ == "__main__":
    for fold in range(5):
        run(fold)
