import pathlib
import gc
import datetime
from typing import Tuple, Type
import re
import os
import time
import warnings

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype


def reduce_mem_usage(df, use_float16=False, cols_exclude=[]):
    """
    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.

    Original code from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage by @gemartin
    Modified to support timestamp type, categorical type
    Modified to add option to use float16
    """

    start_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    cols = [c for c in df.columns if c not in cols_exclude]
    print(cols)

    for col in cols:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            continue
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df


def df_sample_random_buildings(df, b_col, n=500):
    '''Generate a sample of the dataset based
    on randomly selected ruts
    '''
    np.random.seed(42)
    randbuilding = np.random.choice(
        df[b_col].unique(),
        size=n,
        replace=False
    )
    return df[df[b_col].isin(randbuilding)], randbuilding

def print_full(df, num_rows=100):
    '''Print the first num_rows rows of dataframe in full

    Resets display options back to default after printing
    '''
    pd.set_option('display.max_rows', len(df))
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', -1)
    display(df.iloc[0:num_rows])
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')

def add_datepart(
    df, fldnames, datetimeformat,
    drop=True, time=False, errors="raise"
):
    if isinstance(fldnames, str):
        fldnames = [fldnames]
    for fldname in fldnames:
        fld = df[fldname]
        fld_dtype = fld.dtype
        if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
            fld_dtype = np.datetime64

        if not np.issubdtype(fld_dtype, np.datetime64):
            df[fldname + '_orig'] = df[fldname].copy()
            df[fldname] = fld = pd.to_datetime(
                fld, format=datetimeformat, errors=errors)
        targ_pre = re.sub('[Dd]ate$', '', fldname)
        attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
                'Is_month_end', 'Is_month_start', 'Is_quarter_end',
                'Is_quarter_start', 'Is_year_end', 'Is_year_start']
        if time:
            attr = attr + ['Hour', 'Minute', 'Second']
        for n in attr:
            df[targ_pre + n] = getattr(fld.dt, n.lower())
        df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
        if drop:
            df.drop(fldname, axis=1, inplace=True)


def rolling_stat(
    df, main_date_col, groupby_cols, cols_rol,
    period, stat_f
):
    '''
    Example usage:
    main_date_col = 'timestamp'
    groupby_cols = ['site_id']
    cols_rol = ['air_temperature', 'dew_temperature']
    period = 24
    stat_f = np.mean

    rolling_stat(df, main_date_col, groupby_cols, cols_rol,
    period, stat_f)
    '''
    df = df.set_index(main_date_col)
    tmp = (
        df[groupby_cols + cols_rol].
        sort_index().
        groupby(groupby_cols).
        rolling(period).
        agg(stat_f)
    )
    tmp = tmp.drop(groupby_cols, 1)
    tmp = tmp.reset_index()

    return tmp
