#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:21:08 2020

@author: dorian
"""
import os
import problem
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression,Lasso, Ridge, ElasticNet, LassoLars, BayesianRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
import xgboost as xgb
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

def _todense(X):
    X = X.copy()
    return X.todense()

#Loading  training data
X, y = problem.get_train_data()

#############################################Preprocessing training data#####################################
data_merger = FunctionTransformer(_merge_external_data)
date_encoder = FunctionTransformer(_encode_dates)

dense_matrix = FunctionTransformer(_todense)

categorical_encoder_ohe = make_pipeline(
    SimpleImputer(strategy="constant", fill_value="missing"),
    OneHotEncoder(handle_unknown="ignore")
)
categorical_cols_ohe = [
     "Arrival", "Departure", "day", "weekday", "holidays", "week", "n_days"
    ]

categorical_encoder_oe = OrdinalEncoder()
categorical_cols_oe = ["Arrival", "Departure"]

numerical_scaler = StandardScaler()
numerical_cols = ["WeeksToDeparture", "std_wtd"]

preprocessor_ohe = make_column_transformer(
    (categorical_encoder_ohe, categorical_cols_ohe),
   )
"""
preprocessor_ohe_std = make_column_transformer(
    (categorical_encoder_ohe, categorical_cols_ohe),
    (numerical_scaler, numerical_cols)
   )

preprocessor_oe_std = make_column_transformer(
    (categorical_encoder_oe, categorical_cols_oe),
    (numerical_scaler, numerical_cols)
   )


preprocessor_oe = make_column_transformer(
    (categorical_encoder_oe, categorical_cols_oe),
   )
"""
#############################################Test All Models#####################################

models = {"lr" : LinearRegression(),
          "lasso" : Lasso(),
          "ridge" : Ridge(),
          "elasticnet": ElasticNet(),
          "lassolars" : LassoLars(),
          "bayridge" : BayesianRidge(),
          "svr" : SVR(),
          "knn" : KNeighborsRegressor(),
          #"gaussianpr" : GaussianProcessRegressor(), mauvais rmse et long à executer
          "decisiontree" : DecisionTreeRegressor(),
          "rf" : RandomForestRegressor(),
          "extratree" : ExtraTreesRegressor(),
          "adaboost" : AdaBoostRegressor(),
          "gradientboost" : GradientBoostingRegressor(),
          "xgb" : xgb.XGBRegressor()
    }




models_todense = ["lassolars","bayridge","gaussianpr"]

pipelines_oe_std = []
pipelines_oe = []
pipelines_ohe_std = []
pipelines_ohe = []

for names in models:
    if names in models_todense:
        #pipelines_oe_std.append((names, make_pipeline(data_merger,date_encoder, preprocessor_oe_std, models[names])))
        #pipelines_ohe_std.append((names, make_pipeline(data_merger,date_encoder, preprocessor_ohe_std,dense_matrix, models[names])))
        #pipelines_oe.append((names, make_pipeline(data_merger,date_encoder, preprocessor_oe, models[names])))
        pipelines_ohe.append((names, make_pipeline(data_merger,date_encoder, preprocessor_ohe,dense_matrix, models[names])))
    else :
        #pipelines_oe_std.append((names, make_pipeline(data_merger,date_encoder, preprocessor_oe_std, models[names])))
        #pipelines_ohe_std.append((names, make_pipeline(data_merger,date_encoder, preprocessor_ohe_std, models[names])))
        #pipelines_oe.append((names, make_pipeline(data_merger,date_encoder, preprocessor_oe, models[names])))
        pipelines_ohe.append((names, make_pipeline(data_merger,date_encoder, preprocessor_ohe, models[names])))

# Evaluation des différents modèle
results_cv_oe_std = dict()
results_cv_oe = dict()
results_cv_ohe_std = dict()
results_cv_ohe = dict()

for name, model in pipelines_ohe:
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    results_cv_ohe[name] = (np.mean(rmse_scores),np.std(rmse_scores))
    print("OHE",str(name), f"RMSE : {np.mean(rmse_scores):.4f} +/- {np.std(rmse_scores):.4f}")

"""
for name, model in pipelines_oe:
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    results_cv_oe[name] = (np.mean(rmse_scores),np.std(rmse_scores))
    print("OE",str(name), f"RMSE : {np.mean(rmse_scores):.4f} +/- {np.std(rmse_scores):.4f}")

for name, model in pipelines_oe_std:
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    results_cv_oe_std[name] = (np.mean(rmse_scores),np.std(rmse_scores))
    print("OE STD",str(name), f"RMSE : {np.mean(rmse_scores):.4f} +/- {np.std(rmse_scores):.4f}")

for name, model in pipelines_ohe_std:
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    results_cv_ohe_std[name] = (np.mean(rmse_scores),np.std(rmse_scores))
    print("OHE STD",str(name), f"RMSE : {np.mean(rmse_scores):.4f} +/- {np.std(rmse_scores):.4f}")

"""

"""
OHE lr RMSE : 0.6272 +/- 0.0169
OHE lasso RMSE : 0.9936 +/- 0.0300
OHE ridge RMSE : 0.6240 +/- 0.0177
OHE elasticnet RMSE : 0.9936 +/- 0.0300
OHE lassolars RMSE : 0.9936 +/- 0.0300
OHE bayridge RMSE : 0.6207 +/- 0.0193
OHE svr RMSE : 0.4425 +/- 0.0295
OHE knn RMSE : 0.8191 +/- 0.0068
OHE decisiontree RMSE : 0.5367 +/- 0.0222
OHE rf RMSE : 0.4256 +/- 0.0219
OHE extratree RMSE : 0.5124 +/- 0.0227
OHE adaboost RMSE : 0.8422 +/- 0.0165
OHE gradientboost RMSE : 0.6359 +/- 0.0292
OHE xgb RMSE : 0.4330 +/- 0.0283
"""