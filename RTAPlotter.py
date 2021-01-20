import numpy as np
from scipy import *
from pylab import *
import math
import cmath
from ctypes import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib as mpl
from decimal import *


##############################################################################
##############################################################################

dx     = 20e-9
dy     = 20e-9
c0     = 3e8
nref   = 1.0
ntra   = 3.42
NFREQs = 500
###########################################################################
###########################   Spectral values #############################
###########################################################################
lam    = np.loadtxt("2_30um_2nm3_PitchRTA.txt",  usecols=(0), skiprows= 1, unpack =True)
Ref    = np.loadtxt("2_30um_2nm3_PitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
Abs    = np.loadtxt("2_30um_2nm3_PitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
Tra    = np.loadtxt("2_30um_2nm3_PitchRTA.txt",  usecols=(2), skiprows= 1, unpack =True)

###########################################################################

###############################################################################
############################## Peak Finding ###################################
# lam = (c0/freq)*1e6
for i in range(100,len(Abs)-1):
	temp = 0.0
	if ((Abs[i] > Abs[i-1]) & (Abs[i] > Abs[i+1]) & (lam[i]>5) & (lam[i]<7.5) & (Abs[i] > temp)):
		temp  = Abs[i]
		peak  = lam[i]
		peakA = Abs[i]
		print("Abs is %f, temp is %f, lam is %f" %(Abs[i], temp, peak))
    	# print(lam[i])

###############################################################################
peakWN = 1/(peak*1e-4)

fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(direction = 'in', width=2, labelsize=20)
plt.ylabel("R, T, A", fontsize = '30')   
plt.xlabel(r"$\rm Wavenumber \ (cm^{-1})$", fontsize = '30')
# plt.xlim(5,14)
plt.ylim(0,1)

plt.plot(1/(lam*1e-4), Ref, label = r'$\rm R_{CubeYee}$', color = "red", linewidth = 2)
plt.plot(1/(lam*1e-4), Tra, label = r'$\rm T_{CubeYee}$', color = "black", linewidth = 2)
plt.plot(1/(lam*1e-4), Abs, label = r'$\rm A_{CubeYee}$', color = "limegreen", linewidth = 2)


plt.scatter(peakWN, peakA, linewidth = 3, s=55,edgecolors = 'black', c='red', zorder = 25, label = r"$ A_{peak = %2.2f \ \mu m}$" %peak)		

plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)   
ax.legend(loc='center left', fontsize='22')
# ax.axhline(y =1, color = 'black')
ax.axvline(x =peakWN, color = 'black')
print("peak is %g"  %peakWN)
# plt.savefig("hBNWorkingResRTA.pdf")
plt.savefig("ThickGamma_230_nm_PitchWN.png")

###############################################################################

fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(direction = 'in', width=2, labelsize=20)
plt.ylabel("R, T, A", fontsize = '30')   
plt.xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '30')
# plt.xlim(5,14)
plt.ylim(0,1)

plt.plot(lam, Ref, label = r'$\rm R_{FDTD \ hBN}$', color = "red", linewidth = 2)
plt.plot(lam, Tra, label = r'$\rm T_{FDTD \ hBN}$', color = "black", linewidth = 2)
plt.plot(lam, Abs, label = r'$\rm A_{FDTD \ hBN}$', color = "limegreen", linewidth = 2)

plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)   
ax.legend(loc='center left', fontsize='22')
# ax.axhline(y =1, color = 'black')
ax.axvline(x =peak, color = 'black')
print("peak is %g"  %peak)
# plt.savefig("hBNWorkingResRTA.pdf")
plt.savefig("ThickGamma_230_nm_Pitch.png")

###############################################################################
peakWN = 1/(peak*1e-4)

fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(direction = 'in', width=2, labelsize=20)
plt.ylabel("R, T, A", fontsize = '30')   
plt.xlabel(r"$\rm Wavenumber \ (cm^{-1})$", fontsize = '30')
plt.xlim(1000,2000)
plt.ylim(0,1)

plt.plot(1/(lam*1e-4), Ref, label = r'$\rm R_{CubeYee}$', color = "red", linewidth = 2)
plt.plot(1/(lam*1e-4), Tra, label = r'$\rm T_{CubeYee}$', color = "black", linewidth = 2)
plt.plot(1/(lam*1e-4), Abs, label = r'$\rm A_{CubeYee}$', color = "limegreen", linewidth = 2)

plt.scatter(peakWN, peakA, linewidth = 3, s=55,edgecolors = 'black', c='red', zorder = 25, label = r"$ A_{peak = %2.2f \ cm^{-1}}$" %peakWN)		

plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)   
ax.legend(loc='center left', fontsize='22')
# ax.axhline(y =1, color = 'black')
ax.axvline(x =peakWN, color = 'black')
print("peak is %g"  %peakWN)
# plt.savefig("hBNWorkingResRTA.pdf")
plt.savefig("ThickGamma_230_nm_PitchWN_1000_2000.png")

###############################################################################

fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(direction = 'in', width=2, labelsize=20)
plt.ylabel("R, T, A", fontsize = '30')   
plt.xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '30')
plt.xlim(5,9)
plt.ylim(0,1)

plt.plot(lam, Ref, label = r'$\rm R_{FDTD \ hBN}$', color = "red", linewidth = 2)
plt.plot(lam, Tra, label = r'$\rm T_{FDTD \ hBN}$', color = "black", linewidth = 2)
plt.plot(lam, Abs, label = r'$\rm A_{FDTD \ hBN}$', color = "limegreen", linewidth = 2)

plt.scatter(peak, peakA, linewidth = 3, s=55,edgecolors = 'black', c='red', zorder = 25, label = r"$ A_{peak = %2.2f \ \mu m}$" %peak)		

plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)   
ax.legend(loc='center left', fontsize='22')
# ax.axhline(y =1, color = 'black')
ax.axvline(x =peak, color = 'black')
print("peak is %g"  %peak)
# plt.savefig("hBNWorkingResRTA.pdf")
plt.savefig("ThickGamma_230_nm_Pitch_5_9.png")
