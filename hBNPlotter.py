import numpy as np
from scipy import *
from pylab import *
import math
import cmath
from ctypes import *
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['agg.path.chunksize'] = 10000
# all units are in m^-1
wp = 1.15136316e16
gamma = 9.79125662e13
eps_inf = 4.87
s = 1.83
omega_nu = 137200
gamma = 3198.7#/(2*math.pi)#700.01702
################################################
eps_infz = 2.95
sz = 0.61
omega_nuz = 74606.285
gammaz = 491.998
################################################
d = 80e-9
imp0 = 376.730313
k00 = linspace (50000, 2500000, 20000)
c0 = 3e8
eps1 = eps_inf + (s*(omega_nu**2))/((omega_nu**2) + 1j*gamma*k00 - k00*k00)
eps1z = eps_infz + (sz*(omega_nuz**2))/((omega_nuz**2) + 1j*gammaz*k00 - k00*k00)


n1 = np.sqrt(eps1)
n1z = np.sqrt(eps1z)

#using the equations in chapter 2.2.r of Macleod
#assuming the impedance of free space cancels out
#assuming the incident media is vacuum with k00 = 0

# unlabled equation on p 38 in Macleod after eqn 2.88 
delta1 = n1*d*k00*2*math.pi
delta1z = n1z*d*k00*2*math.pi


# eqn 2.93 in Macleod
#since we behin at normal incidence eta0 = y0
eta0 = imp0
eta1 = (n1)*imp0
eta2 = imp0
Y =  (eta2*cos(delta1) + 1j*eta1*sin(delta1))/(cos(delta1) + 1j*(eta2/eta1)*sin(delta1))

Rm = abs(((eta0 - Y)/(eta0 + Y))*conj((eta0 - Y)/(eta0 + Y)))

#########################################################
# eta0z = imp0
# eta1z = (n1z)*imp0
# eta2z = imp0
# Yz =  (eta2z*cos(delta1z) + 1j*eta1z*sin(delta1z))/(cos(delta1z) + 1j*(eta2z/eta1z)*sin(delta1z))

# Rmz = abs(((eta0z - Yz)/(eta0z + Yz))*conj((eta0z - Yz)/(eta0z + Yz)))
#########################################################

#Calculating the T

# Calculate (Power) Transmision from the result of problem 5.11 
# from: http://eceweb1.rutgers.edu/~orfanidi/ewa/ch05.pdf
# Note j --> -i Convention in formula below

B = cos(delta1) + 1j*(eta2/eta1)*sin(delta1)
C = eta2*cos(delta1) + 1j*eta1*sin(delta1)
Tm = 4*eta0*real(eta2)/((eta0*B + C)*conj(eta0*B + C))
#################################################################
# k0 = k*1e12
dx = 2e-8
dy = 2e-8
c0 = 3e8


Nx = 5
Ny = 5

NFREQs = 500
nref = 1.0
ntra = 1.0



lam, REF, TRA, ABS   = np.loadtxt("hBN_VacRTA.txt",  usecols=(0, 1, 2, 3), skiprows= 1, unpack =True)

###############################################################################
############################## Peak Finding ###################################
A = (1-(REF+TRA))

for i in range(0,len(A)-1):
    if ((A[i] > A[i-1]) & (A[i] > A[i+1]) & (lam[i]<10)):
    	peak  = 1/(lam[i]*1e-4)
    	peakA = A[i]
    	# print(lam[i])

###############################################################################

############################## Peak Finding ###################################


for i in range(0,len(A)-1):
    if ((REF[i] > REF[i-1]) & (REF[i] > A[i+1]) & (lam[i]<10)):
    	peakRF = 1/(lam[i]*1e-4)
    	peakR  = REF[i]
    	# print(lam[i])

###############################################################################

print("after loop \n")
fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(direction = 'in', width=2, labelsize=20)
plt.ylabel("R", fontsize = '30')   
plt.xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '30')
# plt.xlim(5,14)
plt.ylim(0,1)


plt.plot(k00/100, Rm, label=r'$\rm R_{TM \ hBN}$', color='black', linewidth = 6)
plt.plot(k00/100, Tm, label=r'$\rm T_{TM \ hBN}$', color='black', linewidth = 6)
plt.plot(k00/100, (1-(Rm + Tm)), label=r'$\rm A_{TM \ hBN}$', color='black', linewidth = 6)


plt.plot(1/(lam*1e-4), REF, label = r'$\rm R_{FDTD \ hBN}$', color = "red", linewidth = 2)
plt.plot(1/(lam*1e-4), TRA, label = r'$\rm T_{FDTD \ hBN}$', color = "cyan", linewidth = 2)
plt.plot(1/(lam*1e-4), (ABS), label = r'$\rm A_{FDTD \ hBN}$', color = "limegreen", linewidth = 2)

plt.scatter(peak, peakA, linewidth = 3, s=55,edgecolors = 'black', c='red', zorder = 25, label = r"$ A_{peak = %2.2f \ \mu m}$" %peak)		

# plt.plot(1/(k00*1e-6), Tm, label=r'$\rm T_{TM}$', color='red', linewidth = 2)

# plt.scatter(peak, peakA, s=25,edgecolors = 'black', c='red', zorder = 25, label = r"$ Absorption \ peak = %2.2f \ \mu m$" %peak)		


plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)   
ax.legend(loc='center left', fontsize='22')
# ax.axhline(y =1, color = 'black')
# ax.axvline(x =peak, color = 'black')
print("peak is %g"  %peak)
# plt.savefig("hBNWorkingResRTA.pdf")
plt.savefig("hBNWorkingResRTFullSpec.png")



#######################################################
fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(direction = 'in', width=2, labelsize=20)
plt.ylabel("R, T, A", fontsize = '30')   
plt.xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '30')
plt.xlim(5,9)
plt.ylim(0,1)

plt.plot(1/(k00*1e-6), Rm, label=r'$\rm R_{TM \ hBN}$', color='black', linewidth = 6)
plt.plot(1/(k00*1e-6), Tm, label=r'$\rm T_{TM \ hBN}$', color='black', linewidth = 6)
plt.plot(1/(k00*1e-6), (1-(Rm + Tm)), label=r'$\rm A_{TM \ hBN}$', color='black', linewidth = 6)

plt.plot(lam, REF, label = r'$\rm R_{FDTD \ hBN}$', color = "red", linewidth = 2)
plt.plot(lam, TRA, label = r'$\rm T_{FDTD \ hBN}$', color = "cyan", linewidth = 2)
plt.plot(lam, (ABS), label = r'$\rm A_{FDTD \ hBN}$', color = "limegreen", linewidth = 2)

plt.scatter(peak, peakA, linewidth = 3, s=55,edgecolors = 'black', c='red', zorder = 25, label = r"$ A_{peak = %2.2f \ \mu m}$" %peak)		

plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)   
ax.legend(loc='center left', fontsize='22')
ax.axvline(x =peak, color = 'black')

# ax.axhline(y =1, color = 'black')
# plt.savefig("hBNWorkingResRTA.pdf")
plt.savefig("hBNWorkingResRT5_9_Lam.png")
#######################################################
# plt.show()


#######################################################
fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(direction = 'in', width=2, labelsize=20)
plt.ylabel("R, T, A", fontsize = '30')   
plt.xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '30')
plt.xlim(1000,2000)
plt.ylim(0,1)

# ax.axvline(x =peak, color = 'black', linewidth=4)
# ax.axvline(x =peakRF, color = 'black', linewidth=4)

plt.plot(k00/100, Rm, label=r'$\rm R_{TM \ hBN}$', color='black', linewidth = 6)
plt.plot(k00/100, Tm, label=r'$\rm T_{TM \ hBN}$', color='black', linewidth = 6)
plt.plot(k00/100, (1-(Rm + Tm)), label=r'$\rm A_{TM \ hBN}$', color='black', linewidth = 6)

plt.plot(1/(lam*1e-4), REF, label = r'$\rm R_{FDTD \ hBN}$', color = "red", linewidth = 2)
plt.plot(1/(lam*1e-4), TRA, label = r'$\rm T_{FDTD \ hBN}$', color = "cyan", linewidth = 2)
plt.plot(1/(lam*1e-4), (ABS), label = r'$\rm A_{FDTD \ hBN}$', color = "limegreen", linewidth = 2)

plt.scatter(peak, peakA, linewidth = 3, s=55,edgecolors = 'black', c='limegreen', zorder = 25, label = r"$ A_{peak = %2.2f (cm^{-1})}$" %peak)		
# plt.scatter(peakRF, peakR, linewidth = 3, s=55,edgecolors = 'black', c='red', zorder = 25, label = r"$ R_{peak = %2.2f (cm^{-1})}$" %peakRF)		

plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)   
ax.legend(loc='center left', fontsize='22')

ax.axvline(x =peak, color = 'limegreen')
# ax.axvline(x =peakRF, color = 'red')

# ax.axhline(y =1, color = 'black')
# plt.savefig("hBNWorkingResRTA.pdf")
plt.savefig("hBNWorkingResRT5_9_WN_1Dot.png")
#######################################################
# plt.show()