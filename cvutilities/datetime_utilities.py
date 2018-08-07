import pandas as pd
import dateutil.tz

def convert_to_pandas_utc_naive(datetime_input):
    datetime_pandas = pd.to_datetime(datetime_input)
    if datetime_pandas.tz is None:
        datetime_pandas_utc_naive = datetime_pandas
    else:
        datetime_pandas_utc_naive = datetime_pandas.tz_convert(dateutil.tz.tzutc()).tz_localize(None)
    return datetime_pandas_utc_naive

def convert_to_native_utc_naive(datetime_input):
    datetime_pandas = pd.to_datetime(datetime_input)
    if datetime_pandas.tz is None:
        datetime_pandas_utc_naive = datetime_pandas
    else:
        datetime_pandas_utc_naive = datetime_pandas.tz_convert(dateutil.tz.tzutc()).tz_localize(None)
    datetime_native_utc_naive = datetime_pandas_utc_naive.to_pydatetime()
    return datetime_native_utc_naive
