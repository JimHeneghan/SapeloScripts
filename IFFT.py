import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import math 


# Rx= np.loadtxt("Ref.txt", usecols=(0), skiprows= 1, unpack =True )
# R0 = np.loadtxt("../Vac/Ref/Ref.txt", usecols=(0,), skiprows= 1, unpack =True )
Ix1 = np.loadtxt("../Inc/Inc0.txt", usecols=(0), skiprows= 1, unpack =True )#../StubbyhBN/StubbyVac/Inc/
Ix2 = np.loadtxt("1M/Inc0.txt", usecols=(0), skiprows= 1, unpack =True )
Ix1fft = np.fft.fft(Ix1, len(Ix1))

Ix2fft = np.fft.fft(Ix2, len(Ix2))
c0 = 3e8
ddx = 1e-9
dt = ddx/(2*c0)

fs = 1/(dt*len(Ix1))
f = fs*np.arange(0,len(Ix1))

lam = c0/f

fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(direction = 'in', width=2, labelsize=20)
plt.ylabel("Field Strength", fontsize = '30')   
plt.xlabel(r"$\rm Time Steps$", fontsize = '30')
plt.xlim(0, 50000)


plt.xlim(0, 3)
plt.plot(lam*1e6, Ix2fft, label = "Ricker FFT", color = "black", linewidth = 2)
plt.plot(lam*1e6, Ix1fft, label = "Wiggles FFT", color = "red", linewidth = 2)

plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)   
ax.legend(loc='center right', fontsize='30')
plt.savefig("IfftComp.png")
# plt.show()
plt.clf()


