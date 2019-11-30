#import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn import model_selection, preprocessing, metrics
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from utils import reduce_mem_usage, df_sample_random_buildings, print_full
from utils import add_datepart, rolling_stat

from feature_eng import precip_depth_1_hr_FE, wind_direction_FE

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

imp = SimpleImputer(missing_values=np.nan, strategy='median')


cont_feats = [
        'square_feet',
        'floor_count',
        'air_temperature',
        'dew_temperature',
        'sea_level_pressure',
        'wind_speed',
        'precip_depth_1_hr',
        'precip_depth_1_hr_nan',
        'precip_depth_1_hr_isTrace',
]

cat_feats = [
    'timestampDayofweek',
    'primary_use',
    'year_built',
    'timestampMonth',
#     'timestampWeek',
    'timestampHour',
    'weekend',
    'site_id',
    'building_id',
    'meter'
]

features = cont_feats + cat_feats

MAIN = pathlib.Path('/Users/hudsonps/kaggle/ashrae-energy-prediction/')
SUBMISSIONS_PATH = MAIN / 'submissions'

sample = False

eval_test = True
create_submission = True
submission_name = 'multiple_models.csv'

class Model():
    '''wrapper for some sklearn model'''
    def __init__(self, model):
        self.model = model
        self.fit = model.fit
        self.predict = model.predict

class Model_meter_specific():
    '''more complicated model, in the case when we have a dict of models
    where each model apply for a different kind of meter
    The key should be the number associated with a meter
    The value is the chosen model for that meter'''
    def __init__(self, model_dict):
        self.model_dict = model_dict


    def fit(self, train, y):
        for m in self.model_dict.keys():
            model = self.model_dict[m]
            print(m)
            print(type(m))
            print(type(train))
            print(train.shape)
            print(train.head())
            train_m = train[train['meter'] == m]
            print("***")
            m_ind = train_m.index
            model.fit(train_m, y.loc[m_ind, 'meter_reading'])
            print("A meter model has been trained")
        return model_dict

    def predict(self, test):
        for m in self.model_dict.keys():
            model = self.model_dict[m]
            test_m = test[test['meter'] == m]
            m_ind = train_m.index
            test.loc[m_ind, 'meter_reading'] = model.predict(test_m)
        return test['meter_reading']


#clf = Model(LinearRegression())



def rolling_averages(train, building_metadata, weather_train):
    cols_rol = [
        'air_temperature',
        'dew_temperature',
        'sea_level_pressure',
        'wind_speed'
    ]

    period = 3

    tmp = rolling_stat(
        weather_train, 'timestamp', ['site_id'],
        cols_rol, period, np.mean
    )
    weather_train = weather_train.drop(cols_rol, 1)
    weather_train = weather_train.merge(tmp, how='inner', on=['site_id', 'timestamp'])

    train = train.merge(building_metadata, on='building_id', how='left')
    train = train.merge(weather_train, on=['site_id', 'timestamp'], how='left')

    return train, weather_train

def labels_primary_use(building_metadata):
    '''Generates encodings for the buildings primary use feature'''
    le = LabelEncoder()
    le.fit(building_metadata['primary_use'])
    return le

def feature_engineering(train, weather_train, primary_use_encoder):
    print(train.columns)
    train['square_feet'] = np.log1p(train['square_feet'])

    # Feature engineering: Add datebased features
    # Monday is 0
    # If dayofweek is 5 or 6, then it is a weekend
    # // is quotient division (i.e. 6//5 is equal to 1, 3//5 is 0)
    add_datepart(
        train, 'timestamp', datetimeformat=None,
        drop=False, time=True, errors="raise"
    )

    train['weekend'] = train['timestamp'].dt.weekday // 5

    # precipitation features
    train, precip_m = precip_depth_1_hr_FE(train, m=None)
    train[['precip_depth_1_hr_nan', 'precip_depth_1_hr_isTrace', 'precip_depth_1_hr']]

    # wind features
    train, wind_direction_m = wind_direction_FE(train, m=None)
    train[['wind_direction_nan','wind_direction_sin','wind_direction_cos','wind_direction']]

    # one-hot encoding for building usage
    # Feature engineering: primary_use
    # Apply label encoder
    train['primary_use'] = primary_use_encoder.transform(train['primary_use'])

    # log of meter reading
    if 'meter_reading'in train.columns:
        print("--- Taking the log of the meter reading")
        train['meter_reading'] = np.where(train['meter_reading']>=0,train['meter_reading'],0)
        train['log_meter_reading']= np.log1p(train['meter_reading'])

    return train, weather_train

def train_model(train, model):
    # DNC
    y = train['log_meter_reading']
    train = train[features]


    #when using the imputer to input values, it is possible that
    # a column is completely filled with NAs.
    # This tends to happen for floor_count IF one filters on building_id
    # As a workaround, let's set the floor number to 1 for now

    if train['floor_count'].count() == 0:
        train['floor_count'] = 1
    imp.fit(train)
    train_numpy = pd.DataFrame(imp.transform(train))
    train_numpy.columns = train.columns
    train = train_numpy

    model.fit(train, y)

    return model

def test_predict(test, model):

    print("Starting test set imputation")
    imputed_test = imp.transform(test[features])

    print("Starting the predictions")
    test['meter_reading'] = np.expm1(model.predict(imputed_test))

    print("Clipping predictions for which value is negative")
    test['meter_reading'] = np.clip(test['meter_reading'].values, 0, None)

    print("Saving predictions")
    sample_submission = test[['row_id', 'meter_reading']]

    sample_submission.loc[:,'meter_reading'] = (
        sample_submission.loc[:, 'meter_reading'].
        astype('float32').
        round(2)
    )

    sample_submission.loc[:,'row_id'] = (
        sample_submission.loc[:, 'row_id'].
        astype('int32')
    )
    sample_submission.to_csv(SUBMISSIONS_PATH / submission_name, index=False)


if __name__ == "__main__":
    print("Reading training sets")

    # Reading building metadata
    building_metadata = pd.read_csv(MAIN / 'data' / 'building_metadata.csv')

     # DNC (does not change)
    train = pd.read_csv(MAIN / 'data' / 'train.csv')
    train['timestamp'] = pd.to_datetime(train['timestamp'], infer_datetime_format=True)
    weather_train = pd.read_csv(MAIN / 'data' / 'weather_train.csv')
    weather_train['timestamp'] = pd.to_datetime(weather_train['timestamp'], infer_datetime_format=True)

    print ("Creating encodings for building primary usage")
    primary_use_encoder = labels_primary_use(building_metadata)

    print("Reading test sets")
    test = pd.read_csv(MAIN / 'data' / 'test.csv')
    test['timestamp'] = pd.to_datetime(test['timestamp'])
    weather_test = pd.read_csv(MAIN / 'data' / 'weather_test.csv')
    weather_test['timestamp'] = pd.to_datetime(weather_test['timestamp'])


    # lines added so we can only consider a small sample for testing purposes
   # train = train[train['timestamp'] < '2016-01-02']
   # weather_train = weather_train[weather_train['timestamp'] < '2016-01-02']
   # test = test[test['timestamp'] < '2017-01-02']
   # weather_test = weather_test[weather_test['timestamp'] < '2017-01-02']

    train = train[train['building_id'] == 2]
    test = test[test['building_id'] == 2]

    #train = train[train['meter'].isin([1, 2])]
    #meter_types = train['meter'].unique()
    #test = test[test['meter'].isin([1,2]) ]

  #  dict_model = {
  #  0: LinearRegression(),
  #  1: LinearRegression(),
  #  2: LinearRegression()
   # 3: LinearRegression(),
  #  }

   # clf = Model_meter_specific(dict_model)

    clf = Model(LinearRegression())


    # reduce memory usage
    train = reduce_mem_usage(train, cols_exclude=['timestamp'])

    print("Computing rolling averages")
    train, weather_train = \
    rolling_averages(train, building_metadata, weather_train)

    print("Feature engineering")
    train, weather_train = \
    feature_engineering(train, weather_train, primary_use_encoder)

    print("Training")
    clf = train_model(train, clf)

    if eval_test:
        print("Computing rolling averages for test set")
        test, weather_test = \
        rolling_averages(test, building_metadata, weather_test)

        print("Test Feature engineering")
        test, weather_test = \
            feature_engineering(test, weather_test, primary_use_encoder)

        print("Predicting")
        test_predict(test, clf)

        print(clf)


