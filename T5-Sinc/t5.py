import matplotlib.pyplot as plt
import numpy as np
from numpy import pi as Pi
from scipy.interpolate import interp1d

LINEAR = 'linear'
QUADRATIC = 'quadratic'
CUBIC = 'cubic'
SINC = 'sinc'
legends = ['function', 'sampling']


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


start = 0
end = 30
step = 0.001
f1 = 1.0/8.0 # 1/8.0
t1 = np.arange(start,end,step)
y1 = np.sin(f1 * 2 * Pi * t1)
plt.plot(t1,y1,'r')

fs = -7/8.0 # -7/8.0
ts = 1.0 / abs(fs)
t2 = np.arange(start, end, ts)
y2 = [y1[int((i * ts) / (step))] for i in range(0, len(t2))]

plt.plot(t2, y2, 'ro')

interp(LINEAR, start, end, t2, y2, ts)
#interp(QUADRATIC, start, end, t2, y2, ts)
interp(SINC, start, end, t2, y2, ts)


plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend(legends, loc='best')
plt.show()

