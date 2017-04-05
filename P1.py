import matplotlib.pyplot as plt
import numpy as np 
from numpy import pi as Pi 
from scipy.interpolate import interp1d

start = 0
end = 	Pi
step = 0.001
f1 = 1.0
t1 = np.arange(start,end,step)
y1 = np.sin(f1 * 2 * Pi * t1)
plt.plot(t1,y1,"r")

fs = 1.5
ts = 1.0 / abs(fs)
t2 = np.arange(start, end, ts)
y2 = [y1[int((i * ts) / (step))] for i in range(0, len(t2))]

t2inter = np.linspace(start, end - ts, num=500)

print(len(t2),len(y2),len(t2inter))

f2inter = interp1d(t2, y2, kind='cubic')


#plt.plot(t2, y2, 'go', t2inter, f2inter(t2inter))
plt.plot(t2, y2, 'go')

plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend(['function', 'sampling', 'interpolated'], loc='best')
plt.show()

