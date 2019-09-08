# importing Pandas library
import pandas as pd

def read_data(filename: str) -> pd.DataFrame:
    data = pd.read_csv(filename, low_memory=False)
    if 'csv_27283995' in filename:
        data = data.drop([0,1,2,3,4], axis=0)
    return data

