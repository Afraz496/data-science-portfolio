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

SEED =42
df = pd.read_csv('Data/housing_price/train.csv', index_col=0)
#Read up parse_args(), so that we don't have to manually write our datset name
target_variable='SalePrice' #define your target variable
df.reset_index(inplace=True, drop=True)
drop_features = ['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']
df.drop(columns=drop_features, inplace=True)

train_size=.8
split = math.floor(float(train_size)*len(df)) # 50% train
train_df=df[:split]
actual_test_df=df[split:]

#CAll functions
processor,X_train, X_test, y_train, y_test = Preprocess_data.fit_processor(train_df,target_variable,0.4)
norm_X_train = processor.transform(X_train)
norm_X_test = processor.transform(X_test)

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

#Maybe train xgBoost and get predictions

#train meta model

X_val_meta = np.column_stack(pred_base_model)

meta_model = LinearRegression()
meta_model.fit(X_val_meta, y_test)

#transform actual_test_data , unseen data

X_actual_test,y_actual_test = Prep_data.format_data(actual_test_df,'SalePrice')
norm_X_actual_test= processor.transform(X_actual_test)

rf_pred_new = base_model.predict(norm_X_actual_test)

# Combine the predictions of the base models into a single feature matrix
X_new_meta = np.column_stack(rf_pred_new)

y_new_meta = meta_model.predict(X_new_meta)

print(meta_model.score(X_new_meta,y_actual_test))






