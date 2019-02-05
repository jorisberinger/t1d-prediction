import pandas as pd
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def convert(data: pd.DataFrame) -> pd.DataFrame:
    data = select_columns(data)
    data = create_time_index(data)
    data = convert_glucose_annotation(data)
    return data

def select_columns(data: pd.DataFrame) -> pd.DataFrame:
    logger.info("select columns")
    # Drop all unneccessary columns to save memory
    data = data.drop(['bgValue', 'cgmRawValue', 'cgmAlertValue', 'pumpCgmPredictionValue',
     'basalAnnotation', 'bolusAnnotation', 'bolusCalculationValue',
     'pumpAnnotation', 'exerciseTimeValue', 'exerciseAnnotation', 'heartRateValue',
     'heartRateVariabilityValue', 'stressBalanceValue', 'stressValue', 'sleepValue', 'sleepAnnotation',
     'locationAnnotation', 'mlCgmValue', 'mlAnnotation', 'insulinSensitivityFactor', 'otherAnnotation'], axis=1)
    return data

def create_time_index(data: pd.DataFrame) -> pd.DataFrame:
    logger.info("create time index")
    # create time index from date and time
    data = data.set_index(pd.to_datetime(data['date'] + " " + data['time']))
    return data


def drop_date_and_time(data: pd.DataFrame) -> pd.DataFrame:
    data = data.drop(['time', 'date'], axis=1)
    return data


def convert_glucose_annotation(data: pd.DataFrame) -> pd.DataFrame:
    index = data['glucoseAnnotation'].str.contains('^GLUCOSE_ELEVATION_30', na=False)
    data['glucoseAnnotation'] = data['glucoseAnnotation'][index]
    data['glucoseAnnotation'] = data['glucoseAnnotation'].str[21:].astype(float)
    return data
