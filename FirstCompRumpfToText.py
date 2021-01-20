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

mpl.rcParams['agg.path.chunksize'] = 10000
shadesBlue = ['limegreen', 'springgreen', 'turquoise', 'teal', 'darkblue', 'navy', 'blue', 'dark violet']
ShadesRed  = ['deeppink','fuchsia', 'purple', 'darkviolet', 'blue', 'dodgerblue', 'deepskyblue', 'teal', 'springgreen', 'seagreen', 'limegreen',
 'forestgreen', 'greenyellow','gold', 'orange', 'orangered', 'salmon', 'red', 'darkred', 'lightcoral']
#all units are in m^-1
##############################################################################

##############################################################################
##############################################################################

dx     = 20e-9
dy     = 20e-9
c0     = 3e8
nref   = 1.0
ntra   = 3.42
NFREQs = 500

pitchThou = np.zeros((12,NFREQs), dtype = np.double)
PitchLeng = np.linspace(2.02, 2.78, 20)
PitchLeng = np.asarray(PitchLeng)
# start = np.array([2.02, 2.10])
# PitchLeng = np.insert(PitchLeng, 1,start)
# PitchLeng = np.insert(PitchLeng, 1,2.02)

mul = 0.000
# for i in range (0, 10):
# 	mul = Decimal(mul) + Decimal(0.040)

# 	temp = Decimal(2.22) + Decimal(mul)
# 	temp = "%2.2f" %temp
# 	print(temp)
# 	PitchLeng.append(temp)

print(PitchLeng)

###########################################################################
###########################   Pitch Sweep   ###############################
###########################################################################
REF2_02um    = np.loadtxt("2_02um/Ref/2_02umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_02um      = np.loadtxt("2_02um/Ref/2_02umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)

###########################################################################
REF2_06um    = np.loadtxt("2_06um/Ref/2_06umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_06um      = np.loadtxt("2_06um/Ref/2_06umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)

###########################################################################
REF2_10um    = np.loadtxt("2_10um/Ref/2_10umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_10um      = np.loadtxt("2_10um/Ref/2_10umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
###########################################################################
REF2_14um    = np.loadtxt("2_14um/Ref/2_14umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_14um      = np.loadtxt("2_14um/Ref/2_14umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
###########################################################################
REF2_18um    = np.loadtxt("2_18um/Ref/2_18umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_18um      = np.loadtxt("2_18um/Ref/2_18umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
###########################################################################
REF2_22um    = np.loadtxt("2_22um/Ref/2_22umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_22um      = np.loadtxt("2_22um/Ref/2_22umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
###########################################################################

REF2_26um    = np.loadtxt("2_26um/Ref/2_26umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_26um      = np.loadtxt("2_26um/Ref/2_26umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)

###########################################################################
lam        = np.loadtxt("2_30um/Ref/2_30umPitchRTA.txt",  usecols=(0), skiprows= 1, unpack =True)
REF2_30um  = np.loadtxt("2_30um/Ref/2_30umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_30um    = np.loadtxt("2_30um/Ref/2_30umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)

##########################################################################
REF2_34um = np.loadtxt("2_34um/Ref/2_34umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_34um   = np.loadtxt("2_34um/Ref/2_34umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)

###########################################################################
REF2_38um = np.loadtxt("2_38um/Ref/2_38umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_38um   = np.loadtxt("2_38um/Ref/2_38umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)

###########################################################################
REF2_42um = np.loadtxt("2_42um/Ref/2_42umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_42um   = np.loadtxt("2_42um/Ref/2_42umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)

###########################################################################
REF2_46um = np.loadtxt("2_46um/Ref/2_46umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_46um   = np.loadtxt("2_46um/Ref/2_46umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)

###########################################################################
REF2_50um = np.loadtxt("2_50um/Ref/2_50umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_50um   = np.loadtxt("2_50um/Ref/2_50umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)

###########################################################################
REF2_54um = np.loadtxt("2_54um/Ref/2_54umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_54um   = np.loadtxt("2_54um/Ref/2_54umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)

###########################################################################
REF2_58um = np.loadtxt("2_58um/Ref/2_58umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_58um   = np.loadtxt("2_58um/Ref/2_58umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
# pitchThou[11] = A2_58um
###########################################################################
# REF2_62um = np.loadtxt("2_62um/Ref/2_62umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
# A2_62um   = np.loadtxt("2_62um/Ref/2_62umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)

###########################################################################
REF2_66um = np.loadtxt("2_66um/Ref/2_66umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_66um   = np.loadtxt("2_66um/Ref/2_66umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)

###########################################################################
REF2_70um = np.loadtxt("2_70um/Ref/2_70umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_70um   = np.loadtxt("2_70um/Ref/2_70umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)

###########################################################################
REF2_74um = np.loadtxt("2_74um/Ref/2_74umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_74um   = np.loadtxt("2_74um/Ref/2_74umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)

###########################################################################
REF2_78um = np.loadtxt("2_78um/Ref/2_78umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_78um   = np.loadtxt("2_78um/Ref/2_78umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
###########################################################################

###########################################################################
numPlots = 20

peak  = np.zeros(numPlots, dtype = np.double)
peakA = np.zeros(numPlots, dtype = np.double)

peak2  = np.zeros(numPlots, dtype = np.double)
peakA2 = np.zeros(numPlots, dtype = np.double)
pitchThou = np.zeros((numPlots,NFREQs), dtype = np.double)

pitchThou[0]  = REF2_02um
pitchThou[1]  = REF2_06um
pitchThou[2]  = REF2_10um
pitchThou[3]  = REF2_14um
# pitchThou[4]  = REF2_18um
pitchThou[5]  = REF2_22um
pitchThou[6]  = REF2_26um
pitchThou[7]  = REF2_30um
pitchThou[8]  = REF2_34um
pitchThou[9]  = REF2_38um
pitchThou[10]  = REF2_42um
# pitchThou[11] = REF2_46um
pitchThou[12] = REF2_50um
pitchThou[13] = REF2_54um
# pitchThou[14] = REF2_58um
# pitchThou[15] = REF2_62um
pitchThou[16] = REF2_66um
pitchThou[17] = REF2_70um
pitchThou[18] = REF2_74um
# pitchThou[19] = REF2_78um


###############################################################################
for i in range(0, numPlots):
    for j in range(0,NFREQs-1):
        if ((pitchThou[i,j] < pitchThou[i,j-1]) & (pitchThou[i,j+1] > pitchThou[i,j]) &  ((1/(lam[j]*1e-4) > 1000))):
            peak[i]  = (1/(lam[j]*1e-4))
            # print(j)
            peakA[i] = pitchThou[i,j]
###########################################################################
fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(direction = 'in', width=2, labelsize=20)
plt.ylabel("R", fontsize = '30')   
plt.xlabel(r"$\rm Wavenumber \ (cm^{-1})$", fontsize = '30')
plt.xlim(1000,2000)
plt.ylim(0,1)


plt.plot(1/(lam*1e-4), REF2_02um, label = r'$\rm R_{d_{cc}\ 2.02 \ \mu m}$', color = ShadesRed[0], linewidth = 3)

plt.plot(1/(lam*1e-4), REF2_06um, label = r'$\rm R_{d_{cc}\ 2.06 \ \mu m}$', color = ShadesRed[1], linewidth = 3)

plt.plot(1/(lam*1e-4), REF2_10um, label = r'$\rm R_{d_{cc}\ 2.10 \ \mu m}$', color = ShadesRed[2], linewidth = 3)

plt.plot(1/(lam*1e-4), REF2_14um, label = r'$\rm R_{d_{cc}\ 2.14 \ \mu m}$', color = ShadesRed[3], linewidth = 3)

# plt.plot(1/(lam*1e-4), REF2_18um, label = r'$\rm R_{d_{cc}\ 2.18 \ \mu m}$', color = ShadesRed[4], linewidth = 3)

plt.plot(1/(lam*1e-4), REF2_22um, label = r'$\rm R_{d_{cc}\ 2.22 \ \mu m}$', color = ShadesRed[5], linewidth = 3)

plt.plot(1/(lam*1e-4), REF2_26um, label = r'$\rm R_{d_{cc}\ 2.26 \ \mu m}$', color = ShadesRed[6], linewidth = 3)

plt.plot(1/(lam*1e-4), REF2_30um, label = r'$\rm R_{d_{cc}\ 2.30 \ \mu m}$', color = ShadesRed[7], linewidth = 3)

plt.plot(1/(lam*1e-4), REF2_34um, label = r'$\rm R_{d_{cc}\ 2.34 \ \mu m}$', color = ShadesRed[8], linewidth = 3)

plt.plot(1/(lam*1e-4), REF2_38um, label = r'$\rm R_{d_{cc}\ 2.38 \ \mu m}$', color = ShadesRed[9], linewidth = 3)

plt.plot(1/(lam*1e-4), REF2_42um, label = r'$\rm R_{d_{cc}\ 2.42 \ \mu m}$', color = ShadesRed[10], linewidth = 3)

# plt.plot(1/(lam*1e-4), REF2_46um, label = r'$\rm R_{d_{cc}\ 2.46 \ \mu m}$', color = ShadesRed[11], linewidth = 3)

plt.plot(1/(lam*1e-4), REF2_50um, label = r'$\rm R_{d_{cc}\ 2.50 \ \mu m}$', color = ShadesRed[12], linewidth = 3)

plt.plot(1/(lam*1e-4), REF2_54um, label = r'$\rm R_{d_{cc}\ 2.54 \ \mu m}$', color = ShadesRed[13], linewidth = 3)

# plt.plot(1/(lam*1e-4), REF2_58um, label = r'$\rm R_{d_{cc}\ 2.58 \ \mu m}$', color = ShadesRed[14], linewidth = 3)

# plt.plot(1/(lam*1e-4), REF2_62um, label = r'$\rm R_{d_{cc}\ 2.62 \ \mu m}$', color = ShadesRed[15], linewidth = 3)

plt.plot(1/(lam*1e-4), REF2_66um, label = r'$\rm R_{d_{cc}\ 2.66 \ \mu m}$', color = ShadesRed[16], linewidth = 3)

plt.plot(1/(lam*1e-4), REF2_70um, label = r'$\rm R_{d_{cc}\ 2.70 \ \mu m}$', color = ShadesRed[17], linewidth = 3)

plt.plot(1/(lam*1e-4), REF2_74um, label = r'$\rm R_{d_{cc}\ 2.74 \ \mu m}$', color = ShadesRed[18], linewidth = 3)

# plt.plot(1/(lam*1e-4), REF2_78um, label = r'$\rm R_{d_{cc}\ 2.78 \ \mu m}$', color = ShadesRed[19], linewidth = 3)


for j in range(0,numPlots):
    plt.scatter(peak[j], peakA[j],   linewidth = 2, s=55,edgecolors = 'black', color = ShadesRed[j],  zorder = 25, label = r"$A_{peak = %2.2f}$" %peak[j])       


plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)   
ax.legend(loc='center right', fontsize='12', ncol = 2)
# ax.axhline(y =1, color = 'black')
# ax.axvline(x =0.1, color = 'black')
# plt.savefig("XFAGFilmOnSiDrD3x3Sweep.pdf")
plt.savefig("Dead_Ref5_9lWN.png")
peak  = np.zeros(numPlots, dtype = np.double)
peakA = np.zeros(numPlots, dtype = np.double)
###############################################################################
for i in range(0, numPlots):
	for j in range(0,len(A2_30um)-1):
	    if ((pitchThou[i,j] > pitchThou[i,j-1]) & (pitchThou[i,j+1] < pitchThou[i,j]) & (lam[i] < 4.5)):
	    	peak[i]  = lam[j]
	    	peakA[i] = pitchThou[i,j]
	    	# print(lam[i])
for i in range(0,len(A2_30um)-1):
    if ((A2_30um[i] > A2_30um[i-1]) & (A2_30um[i] > A2_30um[i+1]) & (lam[i] < 7.5)):
    	peak2  = lam[i]
    	peakA2 = A2_30um[i]
###############################################################################
print(peak)
# peak[2] = 6.83

fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(direction = 'in', width=2, labelsize=20)
plt.ylabel("A", fontsize = '30')   
plt.xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '30')
plt.xlim(4.5,9)
plt.ylim(0,0.30)

plt.plot(lam, A2_02um, label = r'$\rm A_{d_{cc}\ 2.02 \ \mu m}$', color = ShadesRed[0], linewidth = 3)

plt.plot(lam, A2_06um, label = r'$\rm A_{d_{cc}\ 2.06 \ \mu m}$', color = ShadesRed[1], linewidth = 3)

plt.plot(lam, A2_10um, label = r'$\rm A_{d_{cc}\ 2.10 \ \mu m}$', color = ShadesRed[2], linewidth = 3)

plt.plot(lam, A2_14um, label = r'$\rm A_{d_{cc}\ 2.14 \ \mu m}$', color = ShadesRed[3], linewidth = 3)

# plt.plot(lam, A2_18um, label = r'$\rm A_{d_{cc}\ 2.18 \ \mu m}$', color = ShadesRed[4], linewidth = 3)

plt.plot(lam, A2_22um, label = r'$\rm A_{d_{cc}\ 2.22 \ \mu m}$', color = ShadesRed[5], linewidth = 3)

plt.plot(lam, A2_26um, label = r'$\rm A_{d_{cc}\ 2.26 \ \mu m}$', color = ShadesRed[6], linewidth = 3)

plt.plot(lam, A2_30um, label = r'$\rm A_{d_{cc}\ 2.30 \ \mu m}$', color = ShadesRed[7], linewidth = 6)

plt.plot(lam, A2_34um, label = r'$\rm A_{d_{cc}\ 2.34 \ \mu m}$', color = ShadesRed[8], linewidth = 6)

plt.plot(lam, A2_38um, label = r'$\rm A_{d_{cc}\ 2.38 \ \mu m}$', color = ShadesRed[9], linewidth = 3)

plt.plot(lam, A2_42um, label = r'$\rm A_{d_{cc}\ 2.42 \ \mu m}$', color = ShadesRed[10], linewidth = 3)

# plt.plot(lam, A2_46um, label = r'$\rm A_{d_{cc}\ 2.46 \ \mu m}$', color = ShadesRed[11], linewidth = 3)

plt.plot(lam, A2_50um, label = r'$\rm A_{d_{cc}\ 2.50 \ \mu m}$', color = ShadesRed[12], linewidth = 3)

plt.plot(lam, A2_54um, label = r'$\rm A_{d_{cc}\ 2.54 \ \mu m}$', color = ShadesRed[13], linewidth = 3)

# plt.plot(lam, A2_58um, label = r'$\rm A_{d_{cc}\ 2.58 \ \mu m}$', color = ShadesRed[14], linewidth = 3)

# plt.plot(lam, A2_62um, label = r'$\rm A_{d_{cc}\ 2.62 \ \mu m}$', color = ShadesRed[15], linewidth = 3)

plt.plot(lam, A2_66um, label = r'$\rm A_{d_{cc}\ 2.66 \ \mu m}$', color = ShadesRed[16], linewidth = 3)

plt.plot(lam, A2_70um, label = r'$\rm A_{d_{cc}\ 2.70 \ \mu m}$', color = ShadesRed[17], linewidth = 3)

plt.plot(lam, A2_74um, label = r'$\rm A_{d_{cc}\ 2.74 \ \mu m}$', color = ShadesRed[18], linewidth = 3)

# plt.plot(lam, A2_78um, label = r'$\rm A_{d_{cc}\ 2.78 \ \mu m}$', color = ShadesRed[19], linewidth = 3)

for j in range(0,numPlots):
	plt.scatter(peak[j], peakA[j],   linewidth = 2, s=55,edgecolors = 'black', color = ShadesRed[j],  zorder = 25, label = r"$A_{peak = %2.2f \ \mu m}$" %peak[j])       
# plt.scatter(peak2, peakA2, linewidth = 3, s=55,edgecolors = 'black', c='blue', zorder = 25, label = r"$A_{ Polariton \ peak = %2.2f \ \mu m}$" %peak2)  
# ax.axvline(x =peak, color = 'red')
# ax.axvline(x =peak2, color = 'blue')


plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)   
ax.legend(loc='lower left', fontsize='12', ncol = 2)
# ax.axhline(y =1, color = 'black')
ax.axvline(x =7.26, color = 'black')

# plt.savefig("XFAGFilmOnSiDrD3x3Sweep.pdf")
plt.savefig("Dead_A_5_9cols.png")

for i in range(0, numPlots):
    for j in range(0,NFREQs-1):
        if ((pitchThou[i,j] < pitchThou[i,j-1]) & (pitchThou[i,j+1] > pitchThou[i,j]) &  ((1/(lam[j]*1e-4) > 1000))):
            peak[i]  = (1/(lam[j]*1e-4))
            # print(j)
            peakA[i] = pitchThou[i,j]

fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(direction = 'in', width=2, labelsize=20)
plt.ylabel(r"$\rm R \ Trough \ (cm^{-1})$", fontsize = '30')   
plt.xlabel(r"$\rm Pitch Length \ (\mu m)$", fontsize = '30')
# plt.xlim(5,9)
plt.ylim(1100,1650)

plt.scatter(PitchLeng ,peak,    linewidth = 3, s=55,edgecolors = 'black', c='red',  zorder = 25) 
# PitchLeng = np.lins
# peak = np.asarray(peak)
# def func(xdat, m, c):
# 	return m*xdat + c
# popt, pcov = curve_fit(func, PitchLeng, peak)
# plt.plot(PitchLeng, func(PitchLeng, *popt), 'b-', label = r"$\rm \lambda_{r} = %.2f d_{cc} + %.2f $" %tuple(popt))


plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)   
# ax.legend(loc='center left', fontsize='22')
# ax.axhline(y =1, color = 'black')
# ax.axvline(x =7.26, color = 'black')

# # plt.savefig("XFAGFilmOnSiDrD3x3Sweep.pdf")
plt.savefig("RToughTrendDead.png")

# PitchLeng = np.linspace(2.02, 2.66, 20)

for i in range (0,20):
	res = 3.4*PitchLeng[i] - 0.93
	print("d_cc = %.2f , res = %.2f" %(PitchLeng[i], res))
# plt.show()
