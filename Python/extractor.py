from Python.Classes import Event
import pandas as pd

def getEvents(data):

    basal = data[data['basalValue'].notnull()]
    basalEvents = basal.apply(
        lambda event:  Event.createTemp(time=event['date']+','+event['time'], dbdt=event['basalValue'], t1=event['date']+','+event['time'], t2=event['date']+','+event['time']), axis=1)

    basalEvents.apply(lambda event: event.toString(), 1)

    bolus = data[data['bolusValue'].notnull()]
    bolusEvents = bolus.apply(
        lambda event: Event.createBolus(time=event['date']+','+event['time'], units=event['bolusValue']), axis=1)

    bolusEvents.apply(lambda event: event.toString(), 1)

    meal = data[data['mealValue'].notnull()]
    mealEvents = meal.apply(
        lambda event: Event.createCarb(time=event['date']+','+event['time'], grams=event['mealValue'], ctype=180), axis=1) # TODO Carbtype

    mealEvents.apply(lambda event: event.toString(), 1)

    # return pd.Series([basalEvents, bolusEvents, mealEvents], index=["basal", "bolus", "meal"])
    return basalEvents.append(bolusEvents).append(mealEvents)
