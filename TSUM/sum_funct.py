import matplotlib.pyplot as plt
import numpy as np 
from numpy import pi as Pi 
from scipy.interpolate import interp1d

def pairGCD(a, b):
	return a if b == 0 else pairGCD(b, a % b)


def pairLCM(a, b):
	return (a * b) / pairGCD(a, b)


def LCM(* values):
	lcm = 1
	for v in values:
		lcm = pairLCM(lcm, v)
	return lcm

legends = []

start = 0
end = 50
step = 0.001
f1 = float(raw_input("Ingrese Frecuencia 1>>"))
t1 = np.arange(start,end,step)
y1 = np.sin(f1 * 2 * Pi * t1)
P1 = 1/f1
plt.plot(t1,y1,"r")
legends.append("Sin funct P1 = {}".format(P1))

f2 = float(raw_input("Ingrese Frecuencia 2>>"))
t2 = np.arange(start,end,step)
y2 = np.sin(f2 * 2 * Pi * t2)
P2 = 1/f2
plt.plot(t2,y2,"g")
legends.append("Sin funct P2 = {}".format(P2))

f3 = float(raw_input("Ingrese Frecuencia 3>>"))
t3 = np.arange(start,end,step)
y3 = np.sin(f3 * 2 * Pi * t3)
P3 = 1/f3
plt.plot(t3,y3,"b")
legends.append("Sin funct P3 = {}".format(P3))

y4 = y1+y2+y3
plt.plot(t3,y4,"y")
legends.append("Sin funct SUMA periodo = {}".format(LCM(P1,P2,P3)))

plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend(legends, loc='best')
plt.show()
