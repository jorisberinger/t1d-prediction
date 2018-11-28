from check import checkCurrent
from Classes import Event, UserData
from readData import read_data
from predict import calculateBG

filename = "../../Data/data0318.csv"

def mainplt():
    uevent = [Event.createBolus(time=10, units=3.8),
              Event.createCarb(time=5, grams=18, ctype=180),
              Event.createTemp(time=0,dbdt=2,t1=10,t2=20)]

    udata = UserData(bginitial= 0, cratio= 2, idur= 3, inputeeffect= 5, sensf = 10, simlength = 4, stats = None)

    calculateBG(uevent, udata, 500)

def main():
    data = read_data(filename)
    data = data[data['date'] == "08.03.18"]

    res = checkCurrent(data)

if __name__ == '__main__':
    main()