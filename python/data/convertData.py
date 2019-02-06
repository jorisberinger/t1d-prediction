from datetime import date, datetime
import numpy as np
import pandas as pd
import logging
import os
logger = logging.getLogger(__name__)


path = os.getenv('T1DPATH', '../')


def convert(data: pd.DataFrame) -> pd.DataFrame:
    data = select_columns(data)
    data = create_time_index(data)
    data = convert_glucose_annotation(data)
    #save_data(data)
    return data

def select_columns(data: pd.DataFrame) -> pd.DataFrame:
    logger.info("select columns")
    # Drop all unneccessary columns to save memory
    data = data.drop(['bgValue', 'cgmRawValue', 'cgmAlertValue', 'pumpCgmPredictionValue',
     'basalAnnotation', 'bolusAnnotation', 'bolusCalculationValue',
     'pumpAnnotation', 'exerciseTimeValue', 'exerciseAnnotation', 'heartRateValue',
     'heartRateVariabilityValue', 'stressBalanceValue', 'stressValue', 'sleepValue', 'sleepAnnotation',
     'locationAnnotation', 'mlCgmValue', 'mlAnnotation', 'insulinSensitivityFactor', 'otherAnnotation'], axis=1, errors='ignore')
    return data

def create_time_index(data: pd.DataFrame) -> pd.DataFrame:
    logger.info("create time index")
    # create time index from date and time
    data = data.set_index(pd.to_datetime(data['date'] + " " + data['time']))
    return data


def drop_date_and_time(data: pd.DataFrame) -> pd.DataFrame:
    data = data.drop(['time', 'date'], axis=1)
    return data

# remove leading chars and convert value to float
def convert_glucose_annotation(data: pd.DataFrame) -> pd.DataFrame:
    index = ~data['glucoseAnnotation'].isna()
    values = data.loc[index]['glucoseAnnotation']
    index2 = values.str.match("^GLUCOSE_ELEVATION_30=-?\d+\.\d+$", na=False)
    values = values.loc[index2].str[21:].astype(float)
    values.loc[~index2] = np.nan
    data['glucoseAnnotation'] = values
    return data

def save_data(data: pd.DataFrame):
    filename = path + "data/csv/data-saved-1.csv"
    with open(filename, 'w') as file:
        file.write(data.to_csv())
        file.close()


def interpolate_cgm(data: pd.DataFrame) -> pd.DataFrame:
    data = data.resample("1T").asfreq()
    data['cgmValue'] = data['cgmValue'].interpolate('quadratic')
    data['cgmValue'] = data['cgmValue'].interpolate('pad')
    data['date'] = data['date'].interpolate('pad')

    return data