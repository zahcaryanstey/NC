"""
Script to split the data set into training testing and validation sets
"""

import pandas as pd
import numpy as np



"""
Function to split a data set into train, validation and test sets.
Function takes as input the path to the data set csv file, and the name of the data set
"""
def train_validation_test(path,dataset):
    df = pd.read_csv(path)
    train, validate, test = \
        np.split(df.sample(frac=1, random_state=42),
                 [int(.8 * len(df)), int(.9 * len(df))]) # use 80 % for training and 10 % for validation and testing
    train.to_csv(dataset+'_train.csv',index=False)
    validate.to_csv(dataset + '_validate.csv', index=False)
    test.to_csv(dataset + '_test.csv', index=False)


