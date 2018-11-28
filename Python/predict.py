import matplotlib.pyplot as plt
import numpy as np
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
    nn = 50  # nn needs to be even
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
    return sensf/cratio*camount*cob(g, ct)


def deltaBGI(g, bolus, sensf, idur):
    return -bolus*sensf*(1-iob(g, idur)/100.0)


def deltaBG(g, sensf, cratio, camount, ct, bolus, idur):
    return deltaBGI(g, bolus, sensf, idur) + deltaBGC(g, sensf, cratio, camount, ct)


def calculateBG(uevent, udata, n):

    simbg = np.array([udata.bginitial] * n)
    simbgc = np.array([0.0] * n)
    simbgi = np.array([0.0] * n)

    simt = udata.simlength * 60
    dt = simt / n
    print("dt", dt)
    for j in range(0, len(uevent)):
        if uevent[j] and uevent[j].etype != "":
            for i in range(0, n):
                if uevent[j].etype == "carb":
                    simbgc[i] = simbgc[i]+deltaBGC(i * dt - uevent[j].time, udata.sensf, udata.cratio, uevent[j].grams, uevent[j].ctype)
                elif uevent[j].etype == "bolus":
                    simbgi[i] = simbgi[i] + deltaBGI(i * dt - uevent[j].time, uevent[j].units, udata.sensf, udata.idur)
                else:
                    simbgi[i] = simbgi[i]+deltatempBGI((i * dt), uevent[j].dbdt, udata.sensf, udata.idur, uevent[j].t1, uevent[j].t2)

    simbg = simbg + simbgc + simbgi
    x = np.array(range(0, n))
    x = x * dt
    return (simbg, simbgc, simbgi, x)
