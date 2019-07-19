from Classes import Event
import PredictionWindow
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
    basal = pd.Series()
    if 'basalValue' in data.columns:
        basal = data[data['basalValue'].notnull()]
        
    if not basal.empty:
        if 'date' in basal.columns:
            basalEvents = basal.apply(
                lambda event:  Event.createTemp(time=event.name, dbdt=event['basalValue'], t1=event['date']+','+event['time'], t2=event['date']+','+event['time']), axis=1
                )
        else:
            basalEvents = basal.apply(
                lambda event:  Event.createTemp(time=event.name, dbdt=event['basalValue'], t1=event.name, t2=event.name), axis=1
                )

        events = events.append(basalEvents)

    bolus = pd.Series()
    if 'bolusValue' in data.columns:
        bolus = data[data['bolusValue'].notnull()]
    
    if not bolus.empty:
        bolusEvents = bolus.apply(
            lambda event: Event.createBolus(time=event.name, units=event['bolusValue']), axis=1)
        events = events.append(bolusEvents)

    meal = pd.Series()
    if 'mealValue' in data.columns:
        meal = data[data['mealValue'].notnull()]
        meal = data[data['mealValue'] > 0]
    
    if not meal.empty:
        if 'absorptionTime' in meal.columns:
            mealEvents = meal.apply(
                lambda event: Event.createCarb(time=event.name, grams=event['mealValue'], ctype=event['absorptionTime']), axis=1) 
        else:
             mealEvents = meal.apply(
                lambda event: Event.createCarb(time=event.name, grams=event['mealValue'], ctype=3*60), axis=1) 

        events = events.append(mealEvents)

    return events


def getBGContinious(data):
    cgm = data[data['cgmValue'].notnull()]
    return cgm
