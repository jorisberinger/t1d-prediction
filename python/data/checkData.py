import pandas as pd
import logging
import numpy as np

from Classes import UserData
logger = logging.getLogger(__name__)


def check_window(window: pd.DataFrame, user_data: UserData) -> bool:
    # check that there is data at every check point
    t = np.arange(0., user_data.simlength * 60 - user_data.predictionlength + 1, 15.)
    t = np.append(t, user_data.simlength * 60.)
    selected = window.loc[t]
    nas = selected['cgmValue'].isna()
    return not nas.any()

