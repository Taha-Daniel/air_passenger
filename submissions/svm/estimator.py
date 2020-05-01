import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OrdinalEncoder
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
    C = 100
    gamma = 0.01
    kernel = 'rbf'

    regressor = SVR(
        C=C, gamma=gamma, kernel=kernel
    )

    pipeline = make_pipeline(date_encoder, preprocessor, regressor)
