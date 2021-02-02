import pandas as pd
import numpy as np 
import joblib

if __name__=='__main__':
    # loading processed test data
    test_df = pd.read_pickle('../input/test_processed.pkl')

    # loading model
    model = joblib.load('../models/rf.bin')

    # prediction on test data
    output = model.predict(test_df)   
    
    # submission file
    sample = pd.read_csv('../input/sample_submission.csv')

    # rescaling 
    sample.SalePrice = np.square(output)

    # saving submisison file 
    sample.to_csv('../submission.csv', index=False)