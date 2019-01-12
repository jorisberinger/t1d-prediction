import logging
import subprocess
import json
import os

logger = logging.getLogger(__name__)

folder = "/t1d/data/input/"
path = folder + "2/"
prepjs = "/autotune/oref0/bin/oref0-autotune-prep.js"
corejs = "/autotune/oref0/bin/oref0-autotune-core.js"
profilejson = folder + "profile.json"
profilepumpjson = folder + "profile.pump.json"


def run_autotune(data):
    logger.info("run autotune")
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    #subprocess.run(["node", prepjs ,  "/autotune/data/input/1/pumphistory-15.12.17.json" ,  profilejson ,  "/autotune/data/input/1/glucose-15.12.17.json" ,  profilepumpjson, ">", preppedjson])
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

    for name, group in grouped:
        logger.debug("read - " + name)
        getSensAndCR(name)

def getSensAndCR(datestring):
    with open(path + "autotune-result-" + datestring + ".json", "r") as file:
        data = json.load(file)
        logger.debug("file read " + datestring)
        cr = data["carb_ratio"]
        sens = data["isfProfile"]["sensitivities"]

        logger.debug("carb ratio: " + str(cr))
        logger.debug("sensitivities: " + str(sens))

        return {"cr" : cr, "sens" : sens}