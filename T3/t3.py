import matplotlib.pyplot as plt
import numpy as np 
from numpy import pi as Pi 
from scipy.interpolate import interp1d

plt.subplot(121)
legends = []

start = 0
end = 0.1
step = 0.0001
f1 = 100.0
t1 = np.arange(start,end,step)
y1 = 3 * np.cos(f1 * Pi * t1)
plt.plot(t1,y1,"r")
legends.append("f")


# A) 

# Frequency is 2*Pi* F1= 100Pi
# 		F1 = 50
# Minimum frequency Fs is 100 Hz

# B) Sampling 200 Hz
# x(n) = 3cos(2*Pi*n*50/200)*n = 3cos(Pi/2)*n
 
fs = 200
ts = 1.0 / abs(fs)
t2 = np.arange(start, end, ts)
#y2 = [y1[int((i * ts) / (step))] for i in range(0, len(t2))]
y2 = [3*np.cos(Pi*x/2) for x in range(len(t2))]


t2inter = np.linspace(start, end - ts * 2, num=5000, endpoint=True)
f2inter = interp1d(t2, y2, kind='cubic')
plt.plot(t2, y2, 'ro', t2inter, f2inter(t2inter))
legends.append("s - 200")
legends.append("int - 200")

# C) Sampling 75 Hz
# x(n) = 3cos(100Pi/75)n 
# x(n) = 3cos(4Pi/3)n
# x(n) = 3cos(2Pi-2Pi/3)n
# x(n) = 3cos(2Pi/3)n
fs = 75
ts = 1.0 / abs(fs)
t2 = np.arange(start, end, ts)
#y2 = [y1[int((i * ts) / (step))] for i in range(0, len(t2))]
y2 = [3*np.cos(2*Pi*x/3) for x in range(len(t2))]

t2inter = np.linspace(start, end - ts * 2, num=5000, endpoint=True)
f2inter = interp1d(t2, y2, kind='cubic')
plt.plot(t2, y2, 'go', t2inter, f2inter(t2inter))
legends.append("s - 75")
legends.append("int - 75")
plt.ylim([-4,4])
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend(legends, loc='best')

legends = []
plt.subplot(122)

# D) 
# F2 = f*Fs = 200*f 
# f = 1/4
# F2 = 200/4 = 50.
f1 = 50.0
t1 = np.arange(start,end,step)
y_200 = 3 * np.cos(f1 * 2 * Pi * t1)
plt.plot(t1,y_200,"r")
legends.append("Ident. samples Fs 200")

# Si la muestramos a f = 200 tenemos los valores de muestreo exactos

fs = 200.0
ts = 1.0 / abs(fs)
t2 = np.arange(start, end, ts)
#y2 = [y1[int((i * ts) / (step))] for i in range(0, len(t2))]
y2 = [3*np.cos(Pi*x/2) for x in range(len(t2))]
plt.plot(t2,y2,"ro")
legends.append("Ident. samples Fs 200")


# F2 = f*Fs = 75*f 
# f = 1/3
# F2 = 75/3 = 25.
f1 = 25.0
t1 = np.arange(start,end,step)
y_75 = 3 * np.cos(f1 * 2 * Pi * t1)
plt.plot(t1,y_75,"g")
legends.append("Ident. samples Fs 75")



# Si la muestramos a f = 75 tenemos los valores de muestreo exactos

fs = 75
ts = 1.0 / abs(fs)
t2 = np.arange(start, end, ts)
#y2 = [y1[int((i * ts) / (step))] for i in range(0, len(t2))]
y2 = [3*np.cos(2*Pi*x/3) for x in range(len(t2))]
plt.plot(t2,y2,"go")
legends.append("Ident. samples Fs 75")
plt.ylim([-4,4])
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend(legends, loc='best')
plt.show()
