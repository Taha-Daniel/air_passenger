import pandas as pd
import os

filepath = os.path.join(
    os.path.dirname(__file__), 'external_data.csv'
)
data_weather = pd.read_csv(filepath, parse_dates=["Date"])
X_weather = data_weather
X_weather = X_weather.rename(
        columns={'Date': 'DateOfDeparture', 'AirPort': 'Arrival'}
    )

holidaypath = os.path.join(
  os.path.dirname(__file__), 'Holiday.csv'
    )
data_holiday = pd.read_csv(holidaypath, parse_dates=["date"])
X_date = data_holiday.rename(
        columns={'date': 'DateOfDeparture'}
    )
X_merged = pd.merge(
        X_weather, X_date, how='left', on=['DateOfDeparture'], sort=False
    )
X_merged.to_csv(path_or_buf=filepath,index=False)