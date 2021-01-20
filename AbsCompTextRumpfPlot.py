import numpy as np
from scipy import *
from pylab import *
import math
import cmath
from ctypes import *
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['agg.path.chunksize'] = 10000
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


###########################################################################
###########################  Pitch = 2.38 um ##############################
###########################################################################

# lam       = np.loadtxt("../../2_38um/Ref/2_38umPitchRTA.txt",  usecols=(0), skiprows= 1, unpack =True)
REF2_38um = np.loadtxt("../../2_38um/Ref/2_38umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
TRA2_38um = np.loadtxt("../../2_38um/Ref/2_38umPitchRTA.txt",  usecols=(2), skiprows= 1, unpack =True)
A2_38um   = np.loadtxt("../../2_38um/Ref/2_38umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)

###########################################################################
###########################  Pitch = 2.42 um ##############################
###########################################################################

REF2_42um = np.loadtxt("../../2_42/Ref/2_44umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
TRA2_42um = np.loadtxt("../../2_42/Ref/2_44umPitchRTA.txt",  usecols=(2), skiprows= 1, unpack =True)
A2_42um   = np.loadtxt("../../2_42/Ref/2_44umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)

###########################################################################
###########################  Pitch = 2.46 um ##############################
###########################################################################

lam       = np.loadtxt("2_38umPitchRTA.txt",  usecols=(0), skiprows= 1, unpack =True)
REF2_46um = np.loadtxt("2_38umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
TRA2_46um = np.loadtxt("2_38umPitchRTA.txt",  usecols=(2), skiprows= 1, unpack =True)
A2_46um   = np.loadtxt("2_38umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
###########################################################################

###############################################################################
############################## Peak Finding ###################################
# for i in range(0,len(REF2_38um)-1):
#     if ((REF2_38um[i] > REF2_38um[i-1]) & (REF2_38um[i] > REF2_38um[i+1])):
#     	peakLR = lam[i]
#     	peakR  = REF2_38um[i]

# for i in range(0,len(A2_38um)-1):
#     if ((A2_38um[i] > A2_38um[i-1]) & (A2_38um[i] > A2_38um[i+1])):
#     	peak  = lam[i]
#     	peakA = A2_38um[i]
#     	# print(lam[i])
# for i in range(0,len(A2_38um)-1):
#     if ((A2_38um[i] > A2_38um[i-1]) & (A2_38um[i] > A2_38um[i+1]) & (lam[i] < 7.5)):
#     	peak2  = lam[i]
#     	peakA2 = A2_38um[i]
###############################################################################



################################################################################
fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(direction = 'in', width=2, labelsize=20)
plt.ylabel("R", fontsize = '30')   
plt.xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '30')
plt.xlim(5,9)
plt.ylim(0,0.3)


plt.plot(lam,   A2_38um, label = r'$\rm A_{Pitch = \ 2.38 \ \mu m}$', color = "red", linewidth = 3)
plt.plot(lam,   A2_42um, label = r'$\rm A_{Pitch = \ 2.42 \ \mu m}$', color = "blue", linewidth = 3)
plt.plot(lam,   A2_46um, label = r'$\rm A_{Pitch = \ 2.46 \ \mu m}$', color = "black", linewidth = 3)


# plt.scatter(peak, peakA,   linewidth = 3, s=55,edgecolors = 'black', c='red',  zorder = 25, label = r"$A_{ Plasmon   \ peak = %2.2f \ \mu m}$" %peak)       
# plt.scatter(peak2, peakA2, linewidth = 3, s=55,edgecolors = 'black', c='blue', zorder = 25, label = r"$A_{ Polariton \ peak = %2.2f \ \mu m}$" %peak2)  

# plt.scatter(peak, peakA, linewidth = 3, s=55,edgecolors = 'black', c='red', zorder = 25, label = r"$ A_{ Full \ Device \ peak = %2.2f \ \mu m}$" %peak)   
# plt.scatter(peakLR, peakR, linewidth = 3, s=55,edgecolors = 'black', c='red', zorder = 25, label = r"$ R_{peak = %2.2f \ \mu m}$" %peakLR)            
# plt.scatter(7.26, A2_38um[348], linewidth = 3, s=55,edgecolors = 'black', c='red', zorder = 25, label = r"$ A_{hBN \ in \ Vacuum \ peak = 7.26 \ \mu m}$")        

plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)   
ax.legend(loc='center left', fontsize='22')
# ax.axhline(y =1, color = 'black')
ax.axvline(x =7.26, color = 'black')
# ax.axvline(x =peak, color = 'red')
# ax.axvline(x =peak2, color = 'blue')

plt.savefig("ASpecCompsFullDev5_9.png")
plt.savefig("ASpecCompsFullDev5_9.pdf")

