#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 't1d-prediction\data'))
	print(os.getcwd())
except:
	pass

#%%
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

#%% [markdown]
# ## Load Data from csv Sheet

#%%
data = pd.read_csv("csv/data_full.csv")
data.head()

#%% [markdown]
# Our data has 28 columns, we only need a few so we remove all unnecessary columns



#%%
to_drop =   [ 'cgmRawValue','cgmAlertValue', 'pumpCgmPredictionValue', 'basalAnnotation', 'bolusCalculationValue',
            'heartRateVariabilityValue', 'stressBalanceValue', 'stressValue', 'sleepValue', 'sleepAnnotation', 'locationAnnotation',
            'mlCgmValue','mlAnnotation','insulinSensitivityFactor','otherAnnotation', 'pumpAnnotation', 'exerciseTimeValue',
            'exerciseAnnotation', 'heartRateValue'
            ]

data.drop(to_drop, inplace=True, axis=1)
data.head()
#%% [markdown]
# We keep 9 columns
# Next we will combine the date and time column to a datetime object

#%%
timeFormat = "%d.%m.%y,%H:%M%z"
timeZone = "+0100"
data["datetimeIndex"] = data.apply(lambda row: datetime.datetime.strptime((row.date + ',' + row.time) + timeZone, timeFormat), axis=1)
data.head()


#%% [markdown]
# ## let's analyze the cgmValue
# First we print the cgmValue over the timeindex to visulize the data
#%% 
plt.plot(data['datetimeIndex'], data['cgmValue']) 

#%%[markdown]
# we see some missing parts, we therefore need to fill them
# we can interpolate the missing values if the gaps are not too large
# we will now calculate the time gaps between two measurements


#%%
cgmtrue = data[data['cgmValue'].notnull()]
cgmtrue_datetime = cgmtrue['datetimeIndex']
deltas = []
gaps = []
for i in range(len(cgmtrue_datetime) - 1):
    delta = (cgmtrue_datetime[i+1] - cgmtrue_datetime[i]).seconds / 60
    if delta > 20:
        gaps.append(cgmtrue_datetime[i])
    deltas.append((cgmtrue_datetime[i+1] - cgmtrue_datetime[i]).seconds / 60)

deltas = pd.Series(deltas)
print(deltas.describe())
deltas.plot.box()


#%% [markdown]
# we see that the mean is around 5 minutes between two measurements
# but in the plot we see that we have some outliers, with the maximum delta of almost a day.
# That is definitely to long too interpolate between these points
# lets look closer at the outliers

#%%
outlier = deltas[deltas > 5]
outlier_counts = outlier.value_counts().sort_index()
outlier_counts.plot.bar()
print(outlier_counts)
print("total " + str(outlier_counts.sum()))

#%% [markdown]
# We have a total of 20160 gaps which are more than 5 minutes
# if we interpolate up to 20 minutes we over most of these

#%%
outlier_more_than_20 = outlier[outlier > 20]
outlier_20_counts = outlier_more_than_20.value_counts().sort_index()
print(outlier_20_counts)
print("total " + str(outlier_20_counts.sum()))

#%% [markdown]
# we then have only 261 gaps left
# we can remove the time frames from our data set

#%%
gaps = pd.Series(gaps)
print(gaps)