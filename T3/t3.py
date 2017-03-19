import matplotlib.pyplot as plt
import numpy as np 
from numpy import pi as Pi 
from scipy.interpolate import interp1d

start = 0
end = 0.1
step = 0.0001
f1 = 100.0
t1 = np.arange(start,end,step)
y1 = 3 * np.cos(f1 * 2 * Pi * t1)
plt.plot(t1,y1,"r")

# A) Minimum frequency is 200 Hz

# B) Sampling 200 Hz
fs = 200
ts = 1.0 / abs(fs)
t2 = np.arange(start, end, ts)
y2 = [y1[int((i * ts) / (step))] for i in range(0, len(t2))]
print(y2)

t2inter = np.linspace(start, end - ts * 2, num=5000, endpoint=True)
f2inter = interp1d(t2, y2, kind='cubic')
plt.plot(t2, y2, 'ro', t2inter, f2inter(t2inter))

# C) Sampling 75 Hz
fs = 75
ts = 1.0 / abs(fs)
t2 = np.arange(start, end, ts)
y2 = [y1[int((i * ts) / (step))] for i in range(0, len(t2))]
print(y2)

t2inter = np.linspace(start, end - ts * 2, num=5000, endpoint=True)
f2inter = interp1d(t2, y2, kind='cubic')
plt.plot(t2, y2, 'go', t2inter, f2inter(t2inter))

# D) 

plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend(['f', 's - 200', 'int - 200', 's - 75', 'int - 75'], loc='best')
plt.show()
