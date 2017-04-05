import matplotlib.pyplot as plt
import numpy as np 
from numpy import pi as Pi 
from scipy.interpolate import interp1d

LINEAR = 'linear'
QUADRATIC = 'quadratic'
CUBIC = 'cubic'
SINC = 'sinc'
legends = []

def inter_scipy(start, end, t2, y2, ts, type):
    t2inter = np.linspace(start, end - ts * 2, num=500)
    f2inter = interp1d(t2, y2, kind=type)
    plt.plot(t2inter, f2inter(t2inter))
    legends.append("In. " + type)


def sinc(y, ts, value):
    sum = 0
    for i in range(len(y)):
        fx = y[i]
        if abs(fx) < 1e-7: continue
        c = Pi * (((1.0 * value) / ts) - i)
        if abs(c) < 1e-7: sum += 1
        else: sum += fx * np.sin(c) / c
    return sum


def inter_sinc(start, end, t2, y2, ts):
    t2inter = np.linspace(start, end, num=500)
    y2inter = [sinc(y2, ts, value) for value in t2inter]
    plt.plot(t2inter, y2inter)
    legends.append("In. " + SINC)


def interp(type, start, end, t2, y2, ts):
    if type in [LINEAR, QUADRATIC, CUBIC]:
        inter_scipy(start, end, t2, y2, ts, type)
    elif type == SINC:
        inter_sinc(start, end, t2, y2, ts)

def pairGCD(a, b):
	return a if b == 0 else pairGCD(b, a % b)


def pairLCM(a, b):
	return (a * b) / pairGCD(a, b)


def LCM(* values):
	lcm = 1
	for v in values:
		lcm = pairLCM(lcm, v)
	return lcm

start = 0
end = 50
step = 0.001
f1 = float(input("Ingrese Frecuencia 1>>"))
t1 = np.arange(start,end,step)
y1 = np.sin(f1 * 2 * Pi * t1)
P1 = 1/f1
plt.plot(t1,y1)
legends.append("Sin funct P1 = {}".format(P1))

f2 = float(input("Ingrese Frecuencia 2>>"))
t2 = np.arange(start,end,step)
y2 = np.sin(f2 * 2 * Pi * t2)
P2 = 1/f2
plt.plot(t2,y2)
legends.append("Sin funct P2 = {}".format(P2))

f3 = float(input("Ingrese Frecuencia 3>>"))
t3 = np.arange(start,end,step)
y3 = np.sin(f3 * 2 * Pi * t3)
P3 = 1/f3
plt.plot(t3,y3)
legends.append("Sin funct P3 = {}".format(P3))

y4 = y1+y2+y3
plt.plot(t3,y4,"y")
legends.append("Sin funct SUMA periodo = {}".format(LCM(P1,P2,P3)))

fs = float(raw_input("Ingrese frecuencia de muestro>>")) # -7/8.0
ts = 1.0 / abs(fs)
t5 = np.arange(start, end, ts)
y5 = [y4[int((i * ts) / (step))] for i in range(0, len(t5))]
#t2inter = np.linspace(start, end - ts * 2, num=500)
#f2inter = interp1d(t2, y2, kind='cubic')
#legends.append("Sin funct SUMA interpolated")
interp(LINEAR, start, end, t5, y5, ts)

plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend(legends, loc='best')
plt.show()
