import sounddevice as sd
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
    for i in range(length):
        sum = y[i * 2] + y[i * 2 + 1]
        difference = y[i * 2] - y[i * 2 + 1]
        output[i] = sum
        #output[length + i] = difference

    return output


fs=8000
fr=8000

'''
print ("Input record frequency")
fs = int(input())
print ("Input reproduce frequency")
fr = int(input())
'''

duration = 3 # seconds
y = sd.rec(duration * fs, samplerate=fs, channels=1,dtype='float64')
print ("Recording Audio")
sd.wait()
print ("Audio recording complete , Play Audio")
sd.play(y, fr)
sd.wait()
print ("Play Audio Complete")

# First plot
plt.subplot(2,2,1)
plt.xlabel('time')
plt.ylabel('amplitude')
plt.plot(y)

freq, F = fourier_transform(fs, y)

plt.subplot(2,2,2)
plt.plot(freq, F, 'r-')
plt.xlabel('freq (Hz)')
plt.ylabel('|Y(freq)|')

#plt.show()
# End first plot


# Start second plot
y2 = haar_transform(np.copy(y))

plt.subplot(2,2,3)
plt.plot(y2,'k-')
plt.xlabel('time')
plt.ylabel('amplitude')

freq, F = fourier_transform(fs, y2)

plt.subplot(2,2,4)
plt.plot(freq, F, 'r-')
plt.xlabel('freq (Hz)')
plt.ylabel('|Y(freq)|')

plt.show()
# End second plot
