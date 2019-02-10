from Classes import Event
import pandas as pd
import logging

logger = logging.getLogger(__name__)

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
