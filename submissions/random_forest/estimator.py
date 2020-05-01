import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer


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

    categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    categorical_cols = [
        "Arrival", "Departure", "year", "month", "day", "weekday", "week", "n_days"
    ]

    preprocessor = make_column_transformer(
        (categorical_encoder, categorical_cols),
    )
    n_estimators = 1400
    min_samples_split = 2
    min_samples_leaf = 2
    max_features = 'auto'
    max_depth = 70
    bootstrap = True
    regressor = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, bootstrap=bootstrap, n_jobs=-1
    )

    return make_pipeline(date_encoder, preprocessor, regressor)
