import pandas as pd 
import numpy as np 
from sklearn import model_selection
if __name__=="__main__":
    df = pd.read_csv('../input/train.csv')
    df['kfold'] = -1
    print(df.shape)
    num_bins = int(np.floor(1+np.log2(len(df))))
    df.loc[:,'bins'] = pd.cut(
        df['SalePrice'], bins=num_bins,labels=False
    )
    print(df.shape)
    kf = model_selection.StratifiedKFold(n_splits=5)
    for f, (t_, v_) in enumerate(kf.split(X=df, y=df.bins.values)):
        df.loc[v_, 'kfold'] = f    
    
    df = df.drop('bins', axis=1)

    df.to_csv('../input/train_fold.csv', index=False)
    print(df.shape)