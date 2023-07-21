import pandas as pd
import numpy as np
from sklearn.model_selection import (
    TimeSeriesSplit,
    cross_val_score,
    cross_validate,
    train_test_split,
)
#Function to prepare X_train, y_train, X_test, y_test
def data_prep(df,target_variable,test_sample=0.2):
    features = df.columns.to_list()
    features.remove(target_variable)
    X = df[features]
    y = df[[target_variable]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_sample, random_state=69)
    return X_train, X_test, y_train, y_test

def format_data(df,target_variable):
    """
    Format the Data into X, y to setup different variables for the Machine Learning pipeline

    Parameters
    ----------
    df : DataFrame
        DataFrame to separate into X,y

    Returns
    -------
    Xy : tuple[numpy, numpy]
        NumPy arrays of features and predictor

    """
    y = df[target_variable]
    y = y.to_numpy()
    col_names = df.columns.tolist()
    col_names.remove(target_variable)
    X = df[col_names]

    return X,y