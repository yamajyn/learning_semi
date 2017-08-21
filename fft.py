# -*- coding: utf-8 -*
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy import signal

plt.close('all')

input_file = u"data.CSV"
(time, data) = np.loadtxt(input_file,unpack=True, delimiter=",",usecols = (0,1))

fs = 10000.0 # サンプリング周波数
f,t,Sxx = signal.spectrogram(data, fs, nperseg=512)

plt.figure()
plt.pcolormesh(t,f,Sxx,vmax=1e-6)
plt.xlim([0,18])
plt.xlabel(u"時間 [sec]")
plt.ylabel(u"周波数 [Hz]")
# plt.colorbar()
plt.show()
