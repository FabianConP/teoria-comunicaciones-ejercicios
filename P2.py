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
    t2inter = np.linspace(start, end - ts, num=500)
    f2inter = interp1d(t2, y2, kind=type)
    plt.plot(t2inter, f2inter(t2inter))
    legends.append("In. " + type)


def sinc(y, ts, value):
    sum = 0
    for i in range(len(y)):
        fx = y[i]
        if abs(fx) < 1e-7: continue
        c = Pi * (((1.0 * value) / ts) - i)
        if abs(c) < 1e-7:
            sum += 1
        else:
            sum += fx * np.sin(c) / c
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
        if y[i] * y[i - 1] < 0:
            cnt += 1
            if cnt == 3:
                return t[i]
    return 1


def maximum_method(start, end, t2, y2, ts):
    t2inter = np.linspace(start, end, num=500)
    y2inter = [sinc(y2, ts, value) for value in t2inter]
    return find_period2(t2inter, y2inter)


def find_sign(y, lim):
    n = min(lim, len(y))
    cntpos = 0
    for i in range(n):
        cntpos += 1 if y[i] > 0 else 0
    return 1 if cntpos * 2 > n else -1


def find_period2(t, y):
    maxim = []
    for i in range(2, len(y)):
        y1, y2, y3 = y[i - 2], y[i - 1], y[i]
        if y2 > y1 and y2 > y3:
            maxim.append(i)
            if len(maxim) == 2:
                a, b = maxim[0], maxim[1]
                sign = find_sign(y, 10)
                return (t[b] - t[a]) * sign
    return 1e20


start = 0
end = Pi * 2
step = 0.001
f1 = 1.0
t1 = np.arange(start, end, step)
y1 = np.sin(f1 * 2 * Pi * t1)
plt.plot(t1, y1, 'r')

fs = 1.5
ts = 1.0 / abs(fs)
t2 = np.arange(start, end, ts)
y2 = [y1[int((i * ts) / (step))] for i in range(0, len(t2))]

plt.plot(t2, y2, 'ro')

# interp(LINEAR, start, end, t2, y2, ts)
# interp(QUADRATIC, start, end, t2, y2, ts)
interp(SINC, start, end, t2, y2, ts)
# interp(CUBIC, start, end, t2, y2, ts)

'''
T = inter_sinc_root(start, end, t2, y2, ts)
print("Raices\nPeriodo: ", T)
print("Frecuencia: ", 1.0 / T)
'''

T = maximum_method(start, end, t2, y2, ts)
print("Maximos\nPeriodo: ", T)
print("Frecuencia: ", 1.0 / T)


plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend(legends, loc='best')
plt.show()


