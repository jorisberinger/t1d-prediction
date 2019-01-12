import math
from datetime import datetime
from typing import Any, Union

import matplotlib.pyplot as plt
import numpy as np
import logging
logger = logging.getLogger(__name__)
from Classes import Event

"""   _td := 360
   _tp := 55
 
func initVars(_td int, _tp int) {
   td = float64(_td)
   tp = float64(_tp)
   //Time constant of exp decay
   tau = tp * (1 - tp/td) / (1 - 2*tp/td)
   //Rise time factor
   a = 2 * tau / td
   //Auxiliary scale factor
   S = 1 / (1 - a + (1+a)*math.Exp(-td/tau))
}
 
 func calculateInsulinOnBoard(t float64, ie float64) float64 {
   //IOB curve IOB(t)
   IOB := 1 - S*(1-a)*((math.Pow(t, 2)/(tau*td*(1-a))-t/tau-1)*math.Exp(-t/tau)+1)

   return ie * IOB
}"""


# Insulin on Board Advanced
# g = time in minutes from bolus event
# idur = insulin duration

def init_vars(sensf, idur):
    # Time constant of exp decay
    tau = sensf * (1 - sensf/ idur) / (1 - 2*sensf/ idur)
    # Rise time factor
    a = 2 * tau /  idur
    # Auxiliary scale factor
    s = 1 / (1 - a + (1 + a) * math.exp(- idur / tau))
    varobject = {}
    varobject['tau'] = tau
    varobject['a'] = a
    varobject['s'] = s
    return varobject


def iob_adv(t, ie, idur, varsobject):
    tau = varsobject['tau']
    a = varsobject['a']
    s = varsobject['s']
    # IOB curve IOB(t)
    IOB = 1 - s * (1 - a) * ((math.pow(t, 2) / (tau*idur*(1-a))-t/tau-1) * math.exp(-t/tau)+1)

    if t < 0:
        return 1
    elif t > idur:
        return 0
    else:
        return  IOB




# Insulin on Board
# g = time in minutes from bolus event
# idur = insulin duration
def iob(g, idur):
    tot = 100
    if g <= 0.0:
        tot = 100
    elif g >= idur * 60.0:
        tot = 0.0
    else :
        if idur == 3:
            tot = -3.203e-7*pow(g,4)+1.354e-4*pow(g,3)-1.759e-2*pow(g,2)+9.255e-2*g+99.951
        elif idur == 4:
            tot = -3.31e-8 * pow(g, 4) + 2.53e-5 * pow(g, 3) - 5.51e-3 * pow(g, 2) - 9.086e-2 * g + 99.95
        elif idur == 5:
            tot = -2.95e-8 * pow(g, 4) + 2.32e-5 * pow(g, 3) - 5.55e-3 * pow(g, 2) + 4.49e-2 * g + 99.3
        elif idur == 6:
            tot = -1.493e-8 * pow(g, 4) + 1.413e-5 * pow(g, 3) - 4.095e-3 * pow(g, 2) + 6.365e-2 * g + 99.7
    return tot


# Simpson rule tot integrate IOB.
def intIOB(x1, x2, idur, g):
    nn = 200  # nn needs to be even
    ii = 1

    # init with first and last terms of simpson series
    dx = (x2-x1)/nn
    integral = iob((g-x1), idur) + iob(g-(x1+nn*dx), idur)

    while ii < nn-2:
        integral = integral + 4 * iob(g - (x1 + ii * dx), idur) + 2 * iob(g - (x1 + (ii + 1) * dx), idur)
        ii = ii + 2

    integral = integral * dx / 3.0

    return integral


# scheiner gi curves fig 7-8 from Think Like a Pancreas, fit with a triangle shaped absorbtion rate curve
# see basic math pdf on repo for details
# g is time in minutes, ct is carb type
def cob(g, ct):
    if g <= 0:
        tot = 0
    elif g >= ct:
        tot = 1
    elif (g > 0) and (g <= ct/2):
        tot = 2.0 / pow(ct, 2) * pow(g, 2)
    else:
        tot = -1.0 + 4.0 / ct * (g - pow(g, 2) / (2.0 * ct))
    return tot


def deltatempBGI(g, dbdt, sensf, idur, t1, t2):
    return -dbdt * sensf * ((t2 - t1) - 1 / 100 * intIOB(t1, t2, idur, g))


def deltaBGC(g, sensf, cratio, camount, ct):
    logger.debug("delta bgc")
    logger.debug(str(g) + "\t" + str(sensf) + "\t" + str(cratio) + "\t" + str(camount) + "\t" + str(ct))
    logger.debug(sensf/cratio*camount*cob(g, ct))
    return sensf/cratio*camount*cob(g, ct)


def deltaBGI(g, bolus, sensf, idur):
    return -bolus*sensf*(1-iob(g, idur)/100.0)

def deltaBGI_adv(g, bolus, sensf, idur, varsobject):
    return - bolus * sensf*( 1- iob_adv(g, bolus, idur, varsobject))



def deltaBG(g, sensf, cratio, camount, ct, bolus, idur):
    return deltaBGI(g, bolus, sensf, idur) + deltaBGC(g, sensf, cratio, camount, ct)


def calculateBGAt(index, uevent, udata):
    varsobject = init_vars(udata.sensf, udata.idur * 60)

    simbg = udata.bginitial
    simbg_adv = udata.bginitial
    simbgc = 0
    simbgi = 0
    simbgi_adv = 0

    dt = 1 # dt must be 1, 1 minute intervals
    i = index
    for j in range(0, len(uevent)):
        if uevent.etype.values[j] != "":
                if uevent.etype.values[j] == "carb":
                    simbgc = simbgc + deltaBGC(i * dt - uevent.time.values[j], udata.sensf, udata.cratio, uevent.grams.values[j], uevent.ctype.values[j])
                elif uevent.etype.values[j] == "bolus":
                    simbgi = simbgi + deltaBGI(i * dt - uevent.time.values[j], uevent.units.values[j], udata.sensf, udata.idur)
                    simbgi_adv = simbgi_adv + deltaBGI_adv(i * dt - uevent.time.values[j], uevent.units.values[j], udata.sensf, udata.idur * 60, varsobject)

    simbg_res = simbg + simbgc + simbgi
    simbg_adv = simbg_adv + simbgc + simbgi_adv
    return [simbg_res, simbg_adv]




def calculateBG(uevent, udata):
    n = udata.simlength * 60
    simbg = np.array([udata.bginitial] * n)
    simbgc = np.array([0.0] * n)
    simbgi = np.array([0.0] * n)
    simbgi_adv = np.array([0.0] * n)
    simbg_adv = np.array([udata.bginitial] * n)

    dt = 1

    varsobject = init_vars(udata.sensf, udata.idur * 60)

    for j in range(0, len(uevent)):
        if uevent.etype.values[j] != "":
            for i in range(0, n):
                if uevent.etype.values[j] == "carb":
                    simbgc[i] = simbgc[i]+deltaBGC(i * dt - uevent.time.values[j], udata.sensf, udata.cratio, uevent.grams.values[j], uevent.ctype.values[j])
                elif uevent.etype.values[j] == "bolus":
                    simbgi[i] = simbgi[i] + deltaBGI(i * dt - uevent.time.values[j], uevent.units.values[j], udata.sensf, udata.idur)
                    simbgi_adv[i] = simbgi_adv[i] + deltaBGI_adv(i * dt - uevent.time.values[j], uevent.units.values[j], udata.sensf, udata.idur * 60, varsobject)
                #else:
                  #  simbgi[i] = simbgi[i]+deltatempBGI((i * dt), uevent.dbdt.values[j], udata.sensf, udata.idur, uevent.t1.values[j], uevent.t2.values[j])

    simbg_res = simbg + simbgc + simbgi
    simbg_adv = simbg_adv + simbgc + simbgi_adv
    x = np.linspace(0,n,n)
    return [simbg_res, simbgc, simbgi, x, simbgi_adv, simbg_adv]

# compare two different IOB functions
def compareIobs(userdata, uevents, filename):

    # init parameters
    n = userdata.simlength * 60  # one time step every minute
    simt = userdata.simlength * 60  # Simulation length in minutes
    dt = simt / n   # time steps
    sensf = userdata.sensf  # Insulin sensitivity factor
    idur = userdata.idur  # insulin durtion in hours

    # init empty arrays for results
    x = np.linspace(0, simt, n)
    simbgi_adv = np.array([userdata.bginitial] * n)
    simbgi = np.array([userdata.bginitial] * n)

    # initialize vars for advanced iob funtion
    varsobject = init_vars(sensf, idur*60)

    # calculate BGI for every timestep
    for j in range(0, len(uevents)):
        event = uevents[j]
        for i in range(0, n):
            simbgi[i] = simbgi[i] + deltaBGI(i * dt - uevents[j].time, event.units, sensf, idur)
            simbgi_adv[i] = simbgi_adv[i] + deltaBGI_adv(i * dt - uevents[j].time, event.units, sensf, idur * 60, varsobject)

    # plot results in compare.png
    plt.plot(x, simbgi, label='standard', alpha=0.7)
    plt.plot(x, simbgi_adv, label='advanced', alpha=0.7)
    for event in uevents:
        plt.plot(event.time, 0, "o", label="bolus, " + str(event.units) + " units")
    plt.legend()
    plt.title("Comparison of IOB functions")
    plt.savefig(filename, dpi=600)

def plotIOBs(userdata):
    # create sample events
    uevents = [Event.createBolus(30, 1), Event.createBolus(180, 1)]
    compareIobs(userdata, uevents, "/t1d/results/compare.png")



