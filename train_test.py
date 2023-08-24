import sys
import os
sys.path.append('Functions/')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.decomposition import PCA
import logging
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import (
    TimeSeriesSplit,
    cross_val_score,
    cross_validate,
    train_test_split,
)
import time
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

import Prep_data 
import Preprocess_data
import math
import yaml

SEED =42

# Load the config file
with open('config/config.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

global target_variable
target_variable = config['predictor_name']

def main():
    input_dir = config['input_dir']
    input_filename = config['input_filename']
    df = pd.read_csv(os.path.join(input_dir, input_filename), index_col=0)
    df.reset_index(inplace=True, drop=True)

    # Drop Features from config file
    drop_features = config['drop_features']
    df.drop(columns=drop_features, inplace=True)

    train_size = config['train_size']
    split = math.floor(float(train_size)*len(df)) 
    train_df=df[:split]
    actual_test_df=df[split:]

    # Preprocess the data
    base_model_split = config['base_model_split']
    processor,X_train, X_test, y_train, y_test = Preprocess_data.fit_processor(train_df,target_variable,base_model_split)
    norm_X_train = processor.transform(X_train)
    norm_X_test = processor.transform(X_test)

    print(norm_X_test.shape)
    print(y_test.shape)

    params = {
        'n_estimators':[10,50,100],
        'max_depth':[50,100,200],
        'min_samples_split':[2,3,4]    
    }

    skf = KFold(n_splits=10, random_state=SEED, shuffle=True)
    base_model = GridSearchCV(RandomForestRegressor(), params, cv=skf)
    base_model.fit(norm_X_train, y_train.values.ravel())

    best_param_rf= base_model.best_params_

    r_score_train = base_model.score(norm_X_train, y_train)
    r_score_test = base_model.score(norm_X_test, y_test)

    pred_base_model= base_model.predict(norm_X_test)


    #train meta model

    X_val_meta = np.column_stack(pred_base_model)

    meta_model = LinearRegression()
    meta_model.fit(X_val_meta, y_test.values.ravel())

    #transform actual_test_data , unseen data

    X_actual_test,y_actual_test = Prep_data.format_data(actual_test_df,target_variable)
    norm_X_actual_test= processor.transform(X_actual_test)

    rf_pred_new = base_model.predict(norm_X_actual_test)

    # Combine the predictions of the base models into a single feature matrix
    X_new_meta = np.column_stack(rf_pred_new)

    y_new_meta = meta_model.predict(X_new_meta)

    print(meta_model.score(X_new_meta,y_actual_test))

if __name__ == "__main__":
    main()

