# importing Pandas library
import pandas as pd



def add(a, b: int):
    return a + b


def read_data(filename):
    data = pd.read_csv(filename, low_memory=False)
    return data

