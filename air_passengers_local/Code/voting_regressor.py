#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 15:27:47 2020

@author: dorian
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import problem
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer


def _merge_external_data(X):
    filepath = os.path.join(
        os.path.dirname(__file__), 'external_data.csv'
    )
    # Make sure that DateOfDeparture is of dtype datetime
    X = X.copy()  # modify a copy of X
    X.loc[:, "DateOfDeparture"] = pd.to_datetime(X['DateOfDeparture'])
    # Parse date to also be of dtype datetime
    data_weather = pd.read_csv(filepath, parse_dates=["DateOfDeparture"])

    X_weather = data_weather[['DateOfDeparture', 'Arrival',
                              'Max TemperatureC', 'Mean VisibilityKm', 'holidays']]

    X_merged = pd.merge(
        X, X_weather, how='left', on=['DateOfDeparture', 'Arrival'], sort=False
    )
    return X_merged


def _encode_dates(X):
    # Make sure that DateOfDeparture is of dtype datetime
    X = X.copy()  # modify a copy of X
    X.loc[:, "DateOfDeparture"] = pd.to_datetime(X['DateOfDeparture'])
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, 'year'] = X['DateOfDeparture'].dt.year
    X.loc[:, 'month'] = X['DateOfDeparture'].dt.month
    X.loc[:, 'day'] = X['DateOfDeparture'].dt.day
    X.loc[:, 'weekday'] = X['DateOfDeparture'].dt.weekday
    X.loc[:, 'week'] = X['DateOfDeparture'].dt.week
    X.loc[:, 'n_days'] = X['DateOfDeparture'].apply(
        lambda date: (date - pd.to_datetime("1970-01-01")).days
    )
    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["DateOfDeparture"])


#Loading  training data
X, y = problem.get_train_data()

#########################################################################################################################
                                            #Voting with OneHotEncoder 
#########################################################################################################################

#############################################Preprocessing training data#####################################

data_merger = FunctionTransformer(_merge_external_data)
date_encoder = FunctionTransformer(_encode_dates)
date_cols = ["DateOfDeparture"]

categorical_encoder = make_pipeline(
    SimpleImputer(strategy="constant", fill_value="missing"),
    OneHotEncoder(handle_unknown="ignore")
)
categorical_cols = [
    "Arrival", "Departure", "day", "weekday", "holidays", "week", "n_days"
]

preprocessor = make_column_transformer(
    (categorical_encoder, categorical_cols)
)

#############################################Find best parameters#############################################

#Best parameters RandomForest
n_estimators_rf =  1400
min_samples_split_rf =  2
min_samples_leaf_rf =  2
max_features_rf = 'auto'
max_depth_rf = 70
bootstrap_rf =  True

#Best parameters SVR
C_svr = 100
gamma_svr = 0.01
kernel_svr = 'rbf'


rf = RandomForestRegressor(n_estimators=n_estimators_rf, max_depth=max_depth_rf, max_features=max_features_rf, min_samples_split = min_samples_split_rf, min_samples_leaf = min_samples_leaf_rf, bootstrap = bootstrap_rf, n_jobs = -1)

svr = SVR(C = C_svr, gamma = gamma_svr, kernel = kernel_svr)


regressor_voting = VotingRegressor(estimators=[ ('rf', rf),
                                               ("svr", svr)])

pipeline = make_pipeline(data_merger,date_encoder,preprocessor, regressor_voting)


scores = cross_val_score(
    pipeline, X, y, cv=5, scoring='neg_mean_squared_error'
)

rmse_scores = np.sqrt(-scores)

print(
    f"RMSE: {np.mean(rmse_scores):.4f} +/- {np.std(rmse_scores):.4f}"
)
#RMSE: 0.3880 +/- 0.0240



