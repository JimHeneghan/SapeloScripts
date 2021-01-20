import numpy as np
from scipy import *
from pylab import *
import math
import matplotlib.pyplot as plt
#all units are in m
eps_inf = 4.87
s = 1.83
omega_nu = 137200
gamma = 3198.7#/(2*math.pi)#700.01702
d = 83e-9
imp0 = 376.730313
k0 = linspace (100000, 200000, 20000)
c0 = 3e8
eps1 = eps_inf + (s*(omega_nu**2))/((omega_nu**2) + 1j*gamma*k0 - k0*k0)

n1 = np.sqrt(eps1)

#using the equations in chapter 2.2.r of Macleod
#assuming the impedance of free space cancels out
#assuming the incident media is vacuum with k0 = 0

# unlabled equation on p 38 in Macleod after eqn 2.88 
delta1 = n1*d*k0*2*math.pi


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

# fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
# #plt.tight_layout(pad=5.4, w_pad=5.5, h_pad=5.4)
# plt.setp(ax.spines.values(), linewidth=2)
# tick_params(direction = 'in', width=2, labelsize=20)
# # ylabel("T", fontsize = '30')   
# ax.set_xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '30')
# ax.legend(loc='upper right', fontsize='10')
# plt.tight_layout()

R = np.loadtxt("Ref.txt", usecols=(0), skiprows= 1, unpack =True )
I = np.loadtxt("../../Vac/Inc/Inc.txt", usecols=(0), skiprows= 1, unpack =True )#../StubbyhBN/StubbyVac/Inc/
T = np.loadtxt("../Trans/Trans.txt", usecols=(0), skiprows= 1, unpack =True )

stopper = len(R)
print(len(R))
c0 = 3e8
ddx = 1e-9
dt = ddx/(2*c0)
print(dt)


R0tail =  np.zeros(stopper)
R0tail[0:stopper] = R[0:stopper]
plt.plot(R0tail)
plt.savefig("Rtail.png")
plt.clf()

I0tail =  np.zeros(stopper)
I0tail[0:stopper] = I[0:stopper]
plt.plot(I0tail)
plt.savefig("Itail.png")
plt.clf()

Ifft = np.fft.fft(I0tail, len(R0tail))
Rfft = np.fft.fft(R0tail, len(R0tail))
Tfft = np.fft.fft(T, len(R0tail))

fs = 1/(dt*len(R0tail))
f = fs*np.arange(0,len(R0tail))
lam = c0/f

fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(direction = 'in', width=2, labelsize=20)
plt.ylabel("T", fontsize = '30')   
plt.xlabel(r"$\rm Wavenumber \ (cm^{-1})$", fontsize = '30')
# ax.legend(loc='upper right', fontsize='10')
plt.xlim(1200,1700)
plt.ylim(0,1)
# print(len(RT))
# print(min(RT[0:1000]))
wn = (1/lam)/100
# plt.plot(Lambda,R, label = r'$\rm R_{My Code:  Correct \ \gamma}$', color = "red", linewidth = 3)
# plt.plot(Lambda,T, label = r'$\rm T_{My Code: Correct \ \gamma}$', color = "black", linewidth = 3)
# plt.plot(Lambda,RT, label = r'$\rm R+T_{My Code: Correct \ \gamma}$', color = "limegreen", linewidth = 3)

plt.plot(wn, (abs(Rfft)/abs(Ifft))**2,  color = "red", linewidth = 3)
plt.plot(wn, (abs(Tfft)/abs(Ifft))**2,  color = "black", linewidth = 3)
plt.plot(wn, (abs(Tfft)/abs(Ifft))**2 + (abs(Rfft)/abs(Ifft))**2,  color = "limegreen", linewidth = 3)


plot(k0, Rm,  color='darkred')
plot(k0, Tm,   color='green')
plot(k0, Rm + Tm,   color='blue')

plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)   
# ax.legend(loc='center left', fontsize='30')
ax.axvline(x = (1/7.295)*10000, color = 'black', linewidth = 2)
plt.savefig("GPUMyCodehBNRealGammaMacConfirmMay2020_WN.pdf")
plt.savefig("GPUMyCodehBNRealGammaMacConfirmMay2020_WN.png")

# plt.show()
