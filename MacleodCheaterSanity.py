import numpy as np
from scipy import *
from pylab import *
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
d = 50e-9
imp0 = 376.730313
# k0 = k*1e12
c0 = 3e8
numpad = 10000000



R = np.loadtxt("ExRefTime.txt", usecols=(0), skiprows= 1, unpack =True )
# ISim = np.loadtxt("../Vac/Inc/Inc.txt", usecols=(0), skiprows= 1, unpack =True )#../StubbyhBN/StubbyVac/Inc/
# T = np.loadtxt("../Trans/Trans.txt", usecols=(0), skiprows= 1000000, unpack =True )

stopper = len(R)
print(len(R))
padlen = stopper + numpad
c0 = 3e8
ddx = 2e-9
dt = ddx/(2*c0)
print(dt)
timeR = np.linspace(0,padlen, padlen)

pR = pad(R, numpad, mode='constant')

Nlam = 600.0;
arg = ((timeR)*math.pi)/(3*Nlam) - 10.0;
arg = arg*arg;
I = np.exp(-0.5*arg)*np.cos(((timeR)*math.pi/Nlam) - 30.0);

Ifft = np.fft.fft(I, padlen)
Rfft = np.fft.fft(pR, padlen)

fs = 1/(dt*padlen)
f = fs*np.arange(0,padlen)
lam = c0/f



fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(direction = 'in', width=2, labelsize=20)
plt.ylabel("R", fontsize = '30')   
plt.xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '30')
plt.xlim(0,10)
plt.ylim(0,2)
plt.plot(lam*1e6, (abs(Rfft)/abs(Ifft))**2, label = r'$\rm R_{FDTD \ Toy}$', color = "black", linewidth = 6)


plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)   
ax.legend(loc='center right', fontsize='22')
ax.axhline(y =1, color = 'black')
plt.savefig("Toy.pdf")
plt.savefig("Toy.png")
plt.clf()
plt.plot(R)
plt.savefig("Time.png")
# plt.show()
