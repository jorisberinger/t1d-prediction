class Event:
    def __init__(self, time, etype, units=None, grams=None, ctype=None, dbdt=None, t1=None, t2=None):
        self.time = time
        self.etype = etype
        self.units = units
        self.grams = grams
        self.ctype = ctype
        self.dbdt = dbdt
        self.t1 = t1
        self.t2 = t2

    @classmethod
    def createBolus(cls, time, units):
        return Event(time=time, etype="bolus", units=units)

    @classmethod
    def createCarb(cls, time, grams, ctype):
        return Event(time=time, etype="carb", grams=grams*12, ctype=ctype)

    @classmethod
    def createTemp(cls, time, dbdt, t1, t2):
        return Event(time=time, etype="tempbasal", dbdt=dbdt, t1=t1, t2=t2)

    def toString(cls):
        print("Event of Type: " + cls.etype + "\tTime " + cls.time)
        if cls.etype == "bolus":
            print("units " + str(cls.units))
        elif cls.etype == "carb":
            print("grams " + str(cls.grams) + "\tCtype " + str(cls.ctype))
        if cls.etype == "tempbasal":
            print("dbdt " + str(cls.dbdt) + "\tt1 " + cls.t1 + "\tt2 " + cls.t2)



class UserData:
    def __init__(self, bginitial, cratio, idur, inputeeffect, sensf, simlength, stats, predictionlength):
        self.bginitial = bginitial
        self.cratio = cratio
        self.idur = idur
        self.inputeeffect = inputeeffect
        self.sensf = sensf
        self.simlength = simlength
        self.predictionlength = predictionlength
        self.stats = stats
        self.basalprofile = None
