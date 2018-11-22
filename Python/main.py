from Python.predict import reloadGraphData
from Python.Classes import Event, UserData
def main():
    uevent = [Event.createBolus(time=10, units=3.8),
              Event.createCarb(time=5, grams=18, ctype=180),
              Event.createTemp(time=0,dbdt=2,t1=10,t2=20)]

    udata = UserData(bginitial= 0, cratio= 2, idur= 3, inputeeffect= 1, sensf = 2, simlength = 4, stats = None)

    reloadGraphData(uevent, udata, 500)



if __name__ == '__main__':
    main()