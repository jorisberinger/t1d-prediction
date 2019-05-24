import logging
import subprocess
import json
import os
import pandas as pd

from autotune import autotune_prep

logger = logging.getLogger(__name__)

path = os.getenv('T1DPATH', '../')
folder = path + "data/input/"


path = folder + "1/"
prepjs = "/autotune/oref0/bin/oref0-autotune-prep.js"
corejs = "/autotune/oref0/bin/oref0-autotune-core.js"
profilejson = folder + "profile.json"
profilepumpjson = folder + "profile.pump.json"





def run_autotune_calc(data):
    logger.info("run autotune")
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    grouped = data.groupby('date')
    for name, group in grouped:
        logger.debug("prep - " + name)
        with open(path + "prepped_glucose-" + name + ".json", "w") as file:
            proc = subprocess.run(["node", prepjs,  path + "pumphistory-" + name + ".json" ,  profilejson ,  path + "glucose-" + name + ".json" ,  profilepumpjson], encoding='utf-8', stdout=subprocess.PIPE)
            file.write(proc.stdout)

    for name, group in grouped:
        logger.debug("result - " + name)
        with open(path + "autotune-result-" + name + ".json", "w") as file:
            proc = subprocess.run(["node", corejs,  path + "prepped_glucose-" + name + ".json" ,  profilejson, profilepumpjson], encoding='utf-8', stdout=subprocess.PIPE)
            file.write(proc.stdout)

    autotune_res = {}
    for name, group in grouped:
        logger.debug("read - " + name)
        res = getSensAndCR(name)
        autotune_res[name] = res
    return autotune_res

def getSensAndCR(datestring):
    logger.info(datestring)
    with open(path + "autotune-result-" + datestring + ".json", "r") as file:
        data = json.load(file)
        logger.debug("file read " + datestring)
        if 'carb_ratio' in data:
            cr = data["carb_ratio"]
        else:
            logger.warning("carb_ratio not found in autotune-result-" + datestring)
            cr = 4
        if 'isfProfile' in data and 'sensitivities' in data['isfProfile']:
            sens = data["isfProfile"]["sensitivities"]
        else:
            logger.warning("isfProfile not found in autotune-result-" + datestring)
            sens = [{"i": 0,"offset": 0,"sensitivity": 38.965,"start": "00:00:00","x": 0},{"i": 10,"offset": 300,"sensitivity": 30, "start": "05:00:00", "x": 1},{"endOffset": 1440,"i": 14,"offset": 420,"sensitivity": 40,"start": "07:00:00","x": 2}]

        if 'basalprofile' in data:
            basal = data["basalprofile"]
        else:
            logger.warning("basalprofile not found in autotune-result-" + datestring)
            basal = [       {          "i": 0,          "minutes": 0,          "rate": 0.868,          "start": "00:00:00",          "untuned": 17       },       {          "i": 1,          "minutes": 60,          "rate": 0.977,          "start": "01:00:00",          "untuned": 12       },       {          "i": 2,          "minutes": 120,          "rate": 1.083,          "start": "02:00:00",          "untuned": 8       },       {          "i": 3,          "minutes": 180,          "rate": 1.196,          "start": "03:00:00",          "untuned": 7       },       {          "i": 4,          "minutes": 240,          "rate": 1.251,          "start": "04:00:00",          "untuned": 8       },       {          "i": 5,          "minutes": 300,          "rate": 1.219,          "start": "05:00:00",          "untuned": 9       },       {          "i": 6,          "minutes": 360,          "rate": 1.256,          "start": "06:00:00",          "untuned": 6       },       {          "i": 7,          "minutes": 420,          "rate": 1.252,          "start": "07:00:00",          "untuned": 9       },       {          "i": 8,          "minutes": 480,          "rate": 1.212,          "start": "08:00:00",          "untuned": 12       },       {          "i": 9,          "minutes": 540,          "rate": 1.109,          "start": "09:00:00",          "untuned": 22       },       {          "i": 10,          "minutes": 600,          "rate": 1.044,          "start": "10:00:00",          "untuned": 30       },       {          "i": 11,          "minutes": 660,          "rate": 1.028,          "start": "11:00:00",          "untuned": 31       },       {          "i": 12,          "minutes": 720,          "rate": 0.99,          "start": "12:00:00",          "untuned": 33       },       {          "i": 13,          "minutes": 780,          "rate": 0.991,          "start": "13:00:00",          "untuned": 30       },       {          "i": 14,          "minutes": 840,          "rate": 1.005,          "start": "14:00:00",          "untuned": 28       },       {          "i": 15,          "minutes": 900,          "rate": 0.925,          "start": "15:00:00",          "untuned": 19       },       {          "i": 16,          "minutes": 960,          "rate": 0.934,          "start": "16:00:00",          "untuned": 22       },       {          "i": 17,          "minutes": 1020,          "rate": 0.927,          "start": "17:00:00",          "untuned": 24       },       {          "i": 18,          "minutes": 1080,          "rate": 0.963,          "start": "18:00:00",          "untuned": 29       },       {          "i": 19,          "minutes": 1140,          "rate": 0.958,          "start": "19:00:00",          "untuned": 31       },       {          "i": 20,          "minutes": 1200,          "rate": 0.937,          "start": "20:00:00",          "untuned": 33       },       {          "i": 21,          "minutes": 1260,          "rate": 0.948,          "start": "21:00:00",          "untuned": 31       },       {          "i": 22,          "minutes": 1320,          "rate": 0.951,          "start": "22:00:00",          "untuned": 28       },       {          "i": 23,          "minutes": 1380,          "rate": 0.944,          "start": "23:00:00",          "untuned": 19       }    ]

        return {"cr": cr, "sens": sens, "basal" : basal}

def getAllSensAndCR(data):
    grouped = data.groupby('date')
    autotune_res = {}
    for name, group in grouped:
        logger.info("read - " + name)
        res = getSensAndCR(name)
        autotune_res[name] = res
    return autotune_res