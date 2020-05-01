#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 15:27:47 2020

@author: dorian
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression,Lasso, Ridge, ElasticNet, LassoCV, RidgeCV, ElasticNetCV
import problem
from sklearn.model_selection import cross_val_score

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

"""
#Preprocessing training data 
date_encoder = FunctionTransformer(_encode_dates)
X = date_encoder.fit_transform(X)

categorical_cols = ["Departure","Arrival", "year", "month", "day", "weekday", "week", "n_days"]
X = pd.get_dummies(X,columns = categorical_cols)

numerical_cols = ["WeeksToDeparture", "std_wtd"]
X[numerical_cols] = StandardScaler().fit_transform(X[numerical_cols])

#Find best parameters
alphas_1 = np.arange(0.0001,0.1,0.0001)
alphas_2 = np.arange(0.1,10,0.1)

lasso_model = Lasso()
lasso_cv = LassoCV(alphas = alphas_1, cv=5).fit(X, y)
#lasso_best_par = lasso_cv.alpha_
lasso_best_par = 0.0003

ridge_model = Ridge()
ridge_cv = RidgeCV(alphas=alphas_2, cv=5).fit(X, y)
#ridge_best_par = ridge_cv.alpha_
ridge_best_par = 5.2

elastic_model = ElasticNet()
elastic_cv = ElasticNetCV(alphas = alphas_1, cv=5).fit(X, y)
#elastic_best_par = elastic_cv.alpha_
elastic_best_par = 0.0004

"""

#Preprocessing training data 
date_encoder = FunctionTransformer(_encode_dates)

categorical_encoder = OneHotEncoder(handle_unknown="ignore")
categorical_cols = [
    "Arrival", "Departure", "year", "month", "day","weekday", "week", "n_days"
    ]

numerical_scaler = StandardScaler()
numerical_cols = ["WeeksToDeparture", "std_wtd"]


preprocessor = make_column_transformer(
    (categorical_encoder, categorical_cols),
    (numerical_scaler, numerical_cols)
   )

#Best paramaters
elastic_best_par = 0.0004
ridge_best_par = 5.2
lasso_best_par = 0.0003

#Régression lineaire
regressor_linear = LinearRegression()

pipeline_linear = make_pipeline(date_encoder, preprocessor, regressor_linear)

scores_linear = cross_val_score(
    pipeline_linear, X, y, cv=5, scoring='neg_mean_squared_error'
)
rmse_scores_linear = np.sqrt(-scores_linear)

print(
    f"RMSE linear: {np.mean(rmse_scores_linear):.4f} +/- {np.std(rmse_scores_linear):.4f}"
)

#Lasso
regressor_lasso = Lasso(alpha = lasso_best_par)

pipeline_lasso = make_pipeline(date_encoder, preprocessor, regressor_lasso)

scores_lasso = cross_val_score(
    pipeline_lasso, X, y, cv=5, scoring='neg_mean_squared_error'
)
rmse_scores_lasso = np.sqrt(-scores_lasso)

print(
    f"RMSE lasso: {np.mean(rmse_scores_lasso):.4f} +/- {np.std(rmse_scores_lasso):.4f}"
)

#Ridge
regressor_ridge = Lasso(alpha = ridge_best_par)

pipeline_ridge = make_pipeline(date_encoder, preprocessor, regressor_ridge)

scores_ridge = cross_val_score(
    pipeline_ridge, X, y, cv=5, scoring='neg_mean_squared_error'
)
rmse_scores_ridge = np.sqrt(-scores_ridge)

print(
    f"RMSE ridge: {np.mean(rmse_scores_ridge):.4f} +/- {np.std(rmse_scores_ridge):.4f}"
)

#ElasticNet
regressor_elastic = Lasso(alpha = elastic_best_par)

pipeline_elastic = make_pipeline(date_encoder, preprocessor, regressor_elastic)

scores_elastic = cross_val_score(
    pipeline_elastic, X, y, cv=5, scoring='neg_mean_squared_error'
)
rmse_scores_elastic = np.sqrt(-scores_elastic)

print(
    f"RMSE elasticnet: {np.mean(rmse_scores_elastic):.4f} +/- {np.std(rmse_scores_elastic):.4f}"
)

