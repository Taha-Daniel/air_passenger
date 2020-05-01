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
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import make_pipeline
import problem
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor

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
                                            #Gradient boosting with onehotencoder
#########################################################################################################################


#############################################Preprocessing training data#####################################

date_encoder = FunctionTransformer(_encode_dates)
date_cols = ["DateOfDeparture"]

categorical_encoder = OneHotEncoder(handle_unknown="ignore")
categorical_cols = [
    "Arrival", "Departure", "year", "month", "day","weekday", "week", "n_days"
    ]


preprocessor = make_column_transformer(
    (categorical_encoder, categorical_cols),
)

#############################################Find best parameters#############################################

parameters = {
    "gradientboostingregressor__learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.2],
    "gradientboostingregressor__min_samples_split": np.linspace(0.1, 0.5, 10),
    "gradientboostingregressor__min_samples_leaf": np.linspace(0.1, 0.5, 10),
    "gradientboostingregressor__max_depth": [int(x) for x in np.linspace(1, 200, num = 10)],
    "gradientboostingregressor__max_features":["log2","sqrt"],
    "gradientboostingregressor__criterion": ["friedman_mse",  "mae"],
    "gradientboostingregressor__subsample": np.arange(0.4,1,0.1),
    "gradientboostingregressor__n_estimators":[int(x) for x in np.linspace(start = 200, stop = 2000, num = 100)]
    }

pipeline_cv_2 = make_pipeline(date_encoder,preprocessor, GradientBoostingRegressor())
cv_gb_2 = RandomizedSearchCV(pipeline_cv_2, parameters, cv=5, n_jobs=-1, n_iter = 100)

cv_gb_2.fit(X, y)
best_parameters_2 = cv_gb_2.best_params_

"""
#############################################Create pipeline with best parameters###############################

#Best parameters
subsample = 0.89
n_estimators = 490
min_samples_split = 0.23
min_samples_leaf = 0.1
max_features = "sqrt"
max_depth = 155
learning_rate = 0.1
criterion = 'mae'

#Gradient Boosting
gradient_boosting_2 = GradientBoostingRegressor(learning_rate=learning_rate,subsample = subsample, n_estimators=n_estimators, 
                                              min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                                              max_features=max_features, max_depth=max_depth, criterion=criterion)


pipeline_2 = make_pipeline(date_encoder,preprocessor, gradient_boosting_2)

scores_2 = cross_val_score(
    pipeline_2, X, y, cv=5, scoring='neg_mean_squared_error'
)
rmse_scores_2 = np.sqrt(-scores_2)

print(
    f"RMSE : {np.mean(rmse_scores_2):.4f} +/- {np.std(rmse_scores_2):.4f}"
)
#RMSE : 0.9805 +/- 0.0353
"""
