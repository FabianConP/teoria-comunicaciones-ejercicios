import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

fs=8000
fr=16000

print ("Input record frequency")
fs = int(input())
print ("Input reproduce frequency")
fr = int(input())

duration = 2 # seconds
myrecording = sd.rec(duration * fs, samplerate=fs, channels=1,dtype='float64')
print ("Recording Audio")
sd.wait()
print ("Audio recording complete , Play Audio")
sd.play(myrecording, fr)
sd.wait()
print ("Play Audio Complete")

plt.plot(myrecording)
plt.show()
