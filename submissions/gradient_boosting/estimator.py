import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.compose import make_column_transformer
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


def get_estimator():
    date_encoder = FunctionTransformer(_encode_dates)
    date_cols = ["DateOfDeparture"]

    categorical_encoder = OrdinalEncoder()
    categorical_cols = ["Arrival", "Departure"]

    preprocessor = make_column_transformer(
        (categorical_encoder, categorical_cols),
    )


    #Best parameters
    subsample = 0.6
    n_estimators = 1327
    min_samples_split = 0.18
    min_samples_leaf = 0.1
    max_features = "log2"
    max_depth = 200
    learning_rate = 0.2
    criterion = 'friedman_mse'

    regressor = GradientBoostingRegressor(learning_rate=learning_rate, subsample=subsample, n_estimators=n_estimators,
                                          min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                                          max_features=max_features, max_depth=max_depth, criterion=criterion)

    return make_pipeline(date_encoder, preprocessor, regressor)
