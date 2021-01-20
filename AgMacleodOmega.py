import numpy as np
from scipy import *
from pylab import *
import math
import matplotlib.pyplot as plt
#all units are in m

wp = 1.15136316e16
gamma = 9.79125662e13
d = 50e-9
imp0 = 376.730313
w = linspace (100e12, 3000e12,  100)
# k0 = k*1e12
c0 = 3e8
eps1 = 1 + (wp*wp)/(w*(1j*gamma-w))

n1 = np.sqrt(eps1)

#using the equations in chapter 2.2.r of Macleod
#assuming the impedance of free space cancels out
#assuming the incident media is vacuum with k0 = 0
# 
# unlabled equation on p 38 in Macleod after eqn 2.88 
delta1 = n1*d*(w/c0)#*2*math.pi


# eqn 2.93 in Macleod
#since we behin at normal incidence eta0 = y0
eta0 = imp0
eta1 = (n1)*imp0
eta2 = imp0
Y =  (eta2*cos(delta1) + 1j*eta1*sin(delta1))/(cos(delta1) + 1j*(eta2/eta1)*sin(delta1))

Rm = abs(((eta0 - Y)/(eta0 + Y))*conj((eta0 - Y)/(eta0 + Y)))

#Calculating the T

# Calculate (Power) Transmision from the result of problem 5.11 
# from: http://eceweb1.rutgers.edu/~orfanidi/ewa/ch05.pdf
# Note j --> -i Convention in formula below

B = cos(delta1) + 1j*(eta2/eta1)*sin(delta1)
C = eta2*cos(delta1) + 1j*eta1*sin(delta1)
Tm = 4*eta0*real(eta2)/((eta0*B + C)*conj(eta0*B + C))

fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
#plt.tight_layout(pad=5.4, w_pad=5.5, h_pad=5.4)
plt.setp(ax.spines.values(), linewidth=2)
tick_params(direction = 'in', width=2, labelsize=20)
# ylabel("T", fontsize = '30')   
ax.set_xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '30')
# ax.legend(loc='upper right', fontsize='10')
# plt.tight_layout()
Rx= np.loadtxt("Ref.txt", usecols=(0), skiprows= 2, unpack =True )
Ix = np.loadtxt("../Vac/Inc/Inc.txt", usecols=(0), skiprows= 1, unpack =True )

Ixfft = np.fft.fft(Ix, len(Rx))
Rxfft = np.fft.fft(Rx, len(Rx))


Ref =  (abs(Rxfft)/abs(Ixfft))**2
c0 = 3e8
ddx = 1e-9
dt = ddx/(2*c0)
fs = 1/(dt*len(Rx))
f = fs*np.arange(0,len(Rx))
Lambda = c0/f


plt.plot(Lambda*1e6,Ref, label = r'$\rm R_{My \ Code:Ag}$', color = "red", linewidth = 3)


xlim(0, 10)
ylim(0, 1.1)
plot((c0*2*math.pi/(w))*1e6, Rm,  label=r'$\rm T_{TM: Ag}$', color='blue')
ax.axhline(y =1, color = 'black')

plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)   
ax.legend(loc='center left', fontsize='30')
# ax.axvline(x = 0.12915, color = 'black', linewidth = 2)
plt.savefig("AgTMPalikCompare.pdf")
plt.savefig("AgTMPalikCompare.png")

# plt.show()
