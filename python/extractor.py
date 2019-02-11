from Classes import Event, PredictionWindow
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def getEventsAsDataFrame(pw: PredictionWindow) -> pd.DataFrame:
    events = getEvents(pw.data)
    if events.empty:
        logger.warning("No events found")
        return pd.DataFrame()
    # convert to dataframe
    events = pd.DataFrame([vars(e) for e in events], index=events.index)
    # Only select events which are not in the prediction timeframe
    events = events[events.index < pw.userData.simlength * 60 - pw.userData.predictionlength]

    # check if events are in prediction frame, if yes, return none because we want only data without these events
    events_in_prediction = events[events.index >= pw.userData.simlength * 60 - pw.userData.predictionlength]
    events_in_prediction = events_in_prediction[events_in_prediction['etype'] != 'tempbasal']
    if not events_in_prediction.empty:
        return pd.DataFrame()
    return events

def getEvents(data: pd.DataFrame) -> pd.Series:
    events = pd.Series()
    basal = data[data['basalValue'].notnull()]
    if not basal.empty:
        basalEvents = basal.apply(
            lambda event:  Event.createTemp(time=event.name, dbdt=event['basalValue'], t1=event['date']+','+event['time'], t2=event['date']+','+event['time']), axis=1)
        events = events.append(basalEvents)

    bolus = data[data['bolusValue'].notnull()]
    if not bolus.empty:
        bolusEvents = bolus.apply(
            lambda event: Event.createBolus(time=event.name, units=event['bolusValue']), axis=1)
        events = events.append(bolusEvents)

    meal = data[data['mealValue'].notnull()]
    if not meal.empty:
        mealEvents = meal.apply(
            lambda event: Event.createCarb(time=event.name, grams=event['mealValue'], ctype=3*60), axis=1) # TODO Carbtype
        events = events.append(mealEvents)

    return events


def getBGContinious(data):
    cgm = data[data['cgmValue'].notnull()]
    return cgm
