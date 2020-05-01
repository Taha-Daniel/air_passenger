#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:21:08 2020

@author: dorian
"""
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
from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import LinearRegression,Lasso, Ridge, ElasticNet, LassoLars, BayesianRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor

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

date_encoder = FunctionTransformer(_encode_dates)

dense_matrix = FunctionTransformer(_todense)

categorical_encoder_ohe = OneHotEncoder(handle_unknown="ignore")
categorical_cols_ohe = [
    "Arrival", "Departure", "year", "month", "day","weekday", "week", "n_days"
    ]

categorical_encoder_oe = OrdinalEncoder()
categorical_cols_oe = ["Arrival", "Departure"]

numerical_scaler = StandardScaler()
numerical_cols = ["WeeksToDeparture", "std_wtd"]


preprocessor_ohe_std = make_column_transformer(
    (categorical_encoder_ohe, categorical_cols_ohe),
    (numerical_scaler, numerical_cols)
   )

preprocessor_oe_std = make_column_transformer(
    (categorical_encoder_oe, categorical_cols_oe),
    (numerical_scaler, numerical_cols)
   )

preprocessor_ohe = make_column_transformer(
    (categorical_encoder_ohe, categorical_cols_ohe),
   )

preprocessor_oe = make_column_transformer(
    (categorical_encoder_oe, categorical_cols_oe),
   )

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
          "gradientboost" : GradientBoostingRegressor()
    }




models_todense = ["lassolars","bayridge","gaussianpr"]

pipelines_oe_std = []
pipelines_oe = []
pipelines_ohe_std = []
pipelines_ohe = []

for names in models:
    if names in models_todense:
        pipelines_oe_std.append((names, make_pipeline(date_encoder, preprocessor_oe_std, models[names])))
        pipelines_ohe_std.append((names, make_pipeline(date_encoder, preprocessor_ohe_std,dense_matrix, models[names])))
        pipelines_oe.append((names, make_pipeline(date_encoder, preprocessor_oe, models[names])))
        pipelines_ohe.append((names, make_pipeline(date_encoder, preprocessor_ohe,dense_matrix, models[names])))
    else :
        pipelines_oe_std.append((names, make_pipeline(date_encoder, preprocessor_oe_std, models[names])))
        pipelines_ohe_std.append((names, make_pipeline(date_encoder, preprocessor_ohe_std, models[names])))
        pipelines_oe.append((names, make_pipeline(date_encoder, preprocessor_oe, models[names])))
        pipelines_ohe.append((names, make_pipeline(date_encoder, preprocessor_ohe, models[names])))

# Evaluation des différents modèle
results_cv_oe_std = dict()
results_cv_oe = dict()
results_cv_ohe_std = dict()
results_cv_ohe = dict()


for name, model in pipelines_oe:
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    results_cv_oe[name] = (np.mean(rmse_scores),np.std(rmse_scores))
    print("OE",str(name), f"RMSE : {np.mean(rmse_scores):.4f} +/- {np.std(rmse_scores):.4f}")

for name, model in pipelines_ohe:
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    results_cv_ohe[name] = (np.mean(rmse_scores),np.std(rmse_scores))
    print("OHE",str(name), f"RMSE : {np.mean(rmse_scores):.4f} +/- {np.std(rmse_scores):.4f}")

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
Best Model :
    SVR en ohe mais presque pareil qu'avec ohe_std
    RF en ohe
    extratree en ohe_std
"""