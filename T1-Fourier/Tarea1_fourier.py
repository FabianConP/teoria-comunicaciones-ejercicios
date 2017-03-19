from numpy import pi as Pi
import numpy as np
import matplotlib.pyplot as plt


def frange(x, y, jump):
    while x < y:
        yield x
        x += jump


def fourier_serie_for_f_x(n_armonicos, x_points, periodo):
    armonicos = [None] * n_armonicos
    fourier = [None] * len(x_points)
    frequencia = [None] * n_armonicos
    for n in range(len(armonicos)):
        armonicos[n] = [None] * len(x_points)
        frequencia[n] = 1.0 / (2 * (n + 1) - 1)
        for x in range(len(x_points)):
            armonicos[n][x] = (np.sin((2 * (n + 1) - 1) * 2 * Pi * x_points[x] / periodo)) / ((2 * (n + 1)) - 1)
    for x in range(len(x_points)):
        fourier[x] = 0
        for n in range(n_armonicos):
            fourier[x] += armonicos[n][x]
        fourier[x] *= (4 / Pi)
    return armonicos, fourier, frequencia


num_armonicos = int(input("Inserte el Numero de armonicos >> "))
periodo = float(input("Ingrese Periodo >>"))
init = -periodo / 2.0
end = periodo / 2.0
step = periodo / 100.0

points = list(frange(init, end, step))
armonicos, fourier, frequencia = fourier_serie_for_f_x(num_armonicos, points, end - init)
plt.subplot(121)

legends = []

# Plot original function
legends.append("Original function")
plt.plot(points, [-1 if x <= 0 else 1 for x in points])

# Plot Fourier series
legends.append("Fourier' Series")
plt.plot(points, fourier, "r")
for k in range(len(armonicos)):
    if k >= 10: break
    plt.plot(points, armonicos[k])
    legends.append('Harmonic ' + str(k + 1))

plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend(legends, loc='best')

plt.subplot(122)
WOs = [(2 * x) - 1 for x in range(1, num_armonicos + 1)]
zeros = [0 for x in range(1, num_armonicos + 1)]
plt.vlines([0] + WOs, [0] + zeros, [0] + frequencia)

plt.grid()
plt.xlabel('Frequency (f)')
plt.ylabel('Amplitude')
plt.legend(['Space Features'], loc='best')

plt.show()
