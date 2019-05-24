import logging
import json
import os
import pandas as pd
import numpy as np
from autotune.autotune_prep import prep_for_autotune
from autotune.autotune import run_autotune_calc
from data.convertData import convert

def run_autotune(filepath : str) -> bool:
    logging.debug("Prep glucose and insulin history for autotune as json")
    # Convert data into pandas Dataframe
    data_frame : pd.DataFrame = convert(pd.read_csv(filepath))
    # Create Glucose history and Insulin Pump History json files, these will be used in autotune
    prep_for_autotune(data_frame)
    # Run autotune to get insulin and carb ratios as json files
    res = run_autotune_calc(data_frame)
    logging.info("---------RES------------")
    logging.info(type(res))
    logging.info(res.keys())
    sens = []
    crs = []
    for k, v in res.items():
        logging.info(v['sens'][0]['sensitivity'])
        sens.append(v['sens'][0]['sensitivity'])
        crs.append(v['cr'])
    
    logging.info(sens)
    logging.info(pd.Series(sens).describe())
    logging.info(pd.Series(crs).describe())
    return res