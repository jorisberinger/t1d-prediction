# importing Pandas library
import pandas as pd



def add(a, b: int):
    return a + b


def read_data():
    data = pd.read_csv("Data/data1.csv", low_memory=False)
    print(data.ndim)
    print(data.shape)


def main():
    print("Start")
    read_data()


if __name__ == '__main__':
    main()

