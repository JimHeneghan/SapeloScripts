import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import math 


Rx= np.loadtxt("Ref.txt", usecols=(0), skiprows= 1, unpack =True )
Ix = np.loadtxt("../Inc/Inc0.txt", usecols=(0), skiprows= 1, unpack =True )#../StubbyhBN/StubbyVac/Inc/

Rx2 = np.loadtxt("1M/Ref.txt", usecols=(0), skiprows= 1, unpack =True )
Ix2 = np.loadtxt("1M/Inc0.txt", usecols=(0), skiprows= 1, unpack =True )#../StubbyhBN/StubbyVac/Inc/


print(len(Rx))

c0 = 3e8
ddx = 1e-9
dt = ddx/(2*c0)
# print(dt)


Ixfft = np.fft.fft(Ix, len(Rx))
Rxfft = np.fft.fft(Rx, len(Rx))

Ix2fft = np.fft.fft(Ix2, len(Rx2))
Rx2fft = np.fft.fft(Rx2, len(Rx2))

fs = 1/(dt*len(Rx))
f = fs*np.arange(0,len(Rx))

lam = c0/f

fs2 = 1/(dt*len(Rx2))
f2 = fs2*np.arange(0,len(Rx2))

lam2 = c0/f2


fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(direction = 'in', width=2, labelsize=20)
plt.ylabel("R", fontsize = '30')   
plt.xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '30')
plt.xlim(0, 10)
plt.ylim(0,1) 
plt.plot(lam2*1e6, (abs(Rx2fft)/abs(Ix2fft))**2, label = "Ricker R", color = "black", linewidth = 4)
plt.plot(lam*1e6, (abs(Rxfft)/abs(Ixfft))**2, label = "Wiggles R", color = "red", linewidth = 2)


plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)   
ax.legend(loc='center right', fontsize='30')
# ax.axvline(x = 7.295, color = 'black', linewidth = 2)
plt.savefig("AgCompR.png")
plt.clf()