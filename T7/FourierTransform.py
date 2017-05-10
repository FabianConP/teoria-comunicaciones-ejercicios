import numpy as np
import matplotlib.pyplot as plt


def fourier_transform(Fs, y):
    n = len(y)  # length of the signal
    k = np.arange(n)
    T = n / Fs
    frq = k / T  # two sides frequency range
    freq = frq[range(int(n / 2))]  # one side frequency range

    Y = np.fft.fft(y) / n  # fft computing and normalization
    Y = Y[range(int(n / 2))]
    return freq, abs(Y)


def haar_transform(y):
    n = len(y)
    output = np.zeros(n)

    length = n >> 1
    #while True:
    for i in range(length):
        sum = y[i * 2] + y[i * 2 + 1]
        difference = y[i * 2] - y[i * 2 + 1]
        output[i] = sum
        #output[length + i] = difference
    '''
        if length == 1:
            return output
        length >>= 1
        for i in range(length):
            y[i] = output[i]
        break
    '''
    return output


Fs = 150                         # sampling rate
Ts = 1.0/Fs                      # sampling interval
t = np.arange(0,1,Ts)            # time vector
ff = 5                           # frequency of the signal
y = np.sin(2 * np.pi * ff * t)

# First plot
plt.subplot(2,2,1)
plt.plot(t,y,'k-')
plt.xlabel('time')
plt.ylabel('amplitude')

freq, F = fourier_transform(Fs, y)

plt.subplot(2,2,2)
plt.plot(freq, F, 'r-')
plt.xlabel('freq (Hz)')
plt.ylabel('|Y(freq)|')

#plt.show()
# End first plot


# Start second plot
y2 = haar_transform(np.copy(y))

plt.subplot(2,2,3)
plt.plot(t,y2,'k-')
plt.xlabel('time')
plt.ylabel('amplitude')

freq, F = fourier_transform(Fs, y2)

plt.subplot(2,2,4)
plt.plot(freq, F, 'r-')
plt.xlabel('freq (Hz)')
plt.ylabel('|Y(freq)|')

plt.show()
# End second plot