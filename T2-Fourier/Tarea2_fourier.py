from numpy import pi as Pi
import numpy as np
import matplotlib.pyplot as plt


def frange(x, y, jump):
    while x < y:
        yield x
        x += jump


def fourier_serie_for_f_x(n_armonicos, x_points, periodo):
    armonicos = [0] * n_armonicos
    fourier = [0] * len(x_points)
    frequencia = [0] * n_armonicos
    armonicos[0] = [2 / Pi for x in range(len(x_points))]
    fourier[0] = 2 / Pi
    for n in range(1, len(armonicos)):
        armonicos[n] = [None] * len(x_points)
        frequencia[n] = 1.0 / ((2 * n) - 1)
        for x in range(len(x_points)):
            armonicos[n][x] = (np.cos(2 * n * x_points[x])) / ((4 * (pow(n, 2))) - 1)
    for x in range(len(x_points)):
        fourier[x] = 0
        for n in range(1, n_armonicos):
            fourier[x] = fourier[x] + armonicos[n][x]
        fourier[x] = (2 / Pi) - (fourier[x] * (4 / Pi))
    return armonicos, fourier, frequencia


num_armonicos = int(input("Inserte el Numero de armonicos >> "))
periodo = float(input("Ingrese Periodo >>"))
init = -periodo / 2.0
end = periodo / 2.0
step = periodo / 100.0

points = [x for x in frange(init, end, step)]
armonicos, fourier, frequencia = fourier_serie_for_f_x(num_armonicos, points, end - init)
plt.subplot(121)
plt.plot(points, fourier, "r")
legends = ["Fourier' Series"]
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
