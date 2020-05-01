import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

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
    n_estimators = 50
    max_features = 'auto'
    max_depth = None
    min_samples_leaf = 1
    random_state = 0
    rf = ExtraTreesRegressor(n_estimators=n_estimators,
                             max_features=max_features,
                             max_depth=max_depth,
                             min_samples_leaf=min_samples_leaf,
                             random_state=random_state)

    return make_pipeline(date_encoder, preprocessor,rf)
