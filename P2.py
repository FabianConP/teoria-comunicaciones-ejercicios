import matplotlib.pyplot as plt
import numpy as np
from numpy import pi as Pi
from scipy.interpolate import interp1d

from scipy import fft

import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt


LINEAR = 'linear'
QUADRATIC = 'quadratic'
CUBIC = 'cubic'
SINC = 'sinc'
legends = ['function', 'sampling']


def inter_scipy(start, end, t2, y2, ts, type):
    t2inter = np.linspace(start, end - ts , num=500)
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


def inter_sinc_root(start, end, t2, y2, ts):
    t2inter = np.linspace(start, end, num=500)
    y2inter = [sinc(y2, ts, value) for value in t2inter]
    return find_period(t2inter, y2inter)

def find_period(t, y):
	cnt = 0
	for i in range(1, len(y)):
		if y[i] * y[i - 1] < 0 :
			cnt += 1
#			print(y[i - 1], y[i], i)
			if cnt % 3 == 0:
				return t[i]
	return 0

def solve_frequency(t1, y1):
	return np.arcsin(y1) / (2 * Pi * t1)

def solve(t,y):
	cnt = 0.0
	sf = 0
	v = []
	for i in range(len(t)):
		sf = solve_frequency(t[i], y[i])
		print sf
		if abs(sf) < 1e5:
			v.append(sf)
#	print("Cnt ", cnt)
	return sum(v) / float(len(v))


start = 0
end = Pi * 8
step = 0.001
f1 = 1.0
t1 = np.arange(start,end,step)
y1 = np.sin(f1 * 2 * Pi * t1)
plt.plot(t1,y1,'r')

fs =  4
ts = 1.0 / abs(fs)
t2 = np.arange(start, end, ts)
y2 = [y1[int((i * ts) / (step))] for i in range(0, len(t2))]

plt.plot(t2, y2, 'ro')

#interp(LINEAR, start, end, t2, y2, ts)
#interp(QUADRATIC, start, end, t2, y2, ts)
interp(SINC, start, end, t2, y2, ts)
#interp(CUBIC, start, end, t2, y2, ts)


T = inter_sinc_root(start, end, t2, y2, ts)
print("Searching roots Periodo: ", T)
print("Frecuencia: ", 1.0 / T)


ind = 1
print(t2[ind], y2[ind])
F = solve_frequency(t2[ind], y2[ind])
print("Solving for Periodo: ", 1.0 /  F)
print("Frecuencia: ", F)



F = solve(t2, y2)
print("Average Solving for Periodo: ", 1.0 /  F)
print("Frecuencia: ", F)


plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend(legends, loc='best')
plt.show()



# Fourier transform
#n = len(y2)
#Y = np.fft.fft(y2)/n              # fft computing and normalization
#Y = Y[range(n/2)]

#plt.plot(fs, abs(Y), 'r-')
#plt.xlabel('freq (Hz)')
#plt.ylabel('|Y(freq)|')

#plt.show() 
