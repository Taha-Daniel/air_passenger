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
                                            #SVM with OneHotEncoder
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

"""
parameters = {'svr__C': [0.1,1, 10, 100, 1000], 
              'svr__gamma': [1,0.1,0.01,0.001],
              'svr__kernel': ['rbf', 'poly', 'sigmoid']}

pipeline_cv = make_pipeline(date_encoder,preprocessor, SVR())
cv_svm = RandomizedSearchCV(pipeline_cv, parameters, cv=5, n_jobs=-1, n_iter = 100)

cv_svm.fit(X, y)
best_parameters = cv_svm.best_params_
"""

#############################################Create pipeline with best parameters##############################

C = 100
gamma = 0.01
kernel= 'rbf'

regressor = SVR(
    C = C, gamma = gamma, kernel = kernel
)

pipeline = make_pipeline(data_merger, date_encoder,preprocessor, regressor)



scores = cross_val_score(
    pipeline, X, y, cv=5, scoring='neg_mean_squared_error'
)
rmse_scores = np.sqrt(-scores)

print(
    f"RMSE: {np.mean(rmse_scores):.4f} +/- {np.std(rmse_scores):.4f}"
)
#RMSE: 0.4278 +/- 0.0283