# importing Pandas library
import pandas as pd


def read_data(filename: str) -> pd.DataFrame:
    data = pd.read_csv(filename, low_memory=False)
    return data

