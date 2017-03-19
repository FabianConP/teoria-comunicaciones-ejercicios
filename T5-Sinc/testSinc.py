import matplotlib.pyplot as plt
import numpy as np
from numpy import pi as Pi

legends = []


def fsinc(x):
    sum = 0
    c = Pi * (x - 1)
    if abs(c) < 1e-7:
        sum += 1
    else:
        sum += np.sin(c)/c
    c = Pi * (x - 3)
    if abs(c) < 1e-7:
        sum += 1
    else:
        sum += 2 * np.sin(c) / c
    return sum


def sinc(y, ts, value):
    sum = 0
    for i in range(len(y)):
        fx = y[i]
        if abs(fx) < 1e-7:
            continue
        c = Pi * (((1.0 * value) / ts) - i)
        if abs(c) < 1e-7:
            sum += 1
        else:
            sum += fx * np.sin(c) / c
    return sum


start = 0
end = 3
step = 0.001
t1 = np.arange(start,end,step)
y1 = [fsinc(x) for x in t1]
#legends.append("sinc manual")
#plt.plot(t1,y1,"r")

x = [0, 1, 2, 3]
y = [0, 1, 0, 2]
ts = 1
y2 = [sinc(y, ts, x) for x in t1]
legends.append("sinc auto")
print(y2)
plt.plot(t1,y2,'c')



plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend(legends, loc='best')
plt.show()

