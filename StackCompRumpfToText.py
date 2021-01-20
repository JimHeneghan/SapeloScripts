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
###########################  Pitch = 2.30 um ##############################
###########################################################################
lam        = np.loadtxt("2_30um/Ref/2_30umPitchRTA.txt",  usecols=(0), skiprows= 1, unpack =True)
REF2_30um  = np.loadtxt("2_30um/Ref/2_30umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_30um    = np.loadtxt("2_30um/Ref/2_30umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)

###########################################################################
# REF2_281um = np.loadtxt("2_281um/Ref/2_281umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)

###########################################################################
REF2_Film = np.loadtxt("FilmStack/Ref/FilmStackRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_Film   = np.loadtxt("FilmStack/Ref/FilmStackRTA.txt",  usecols=(3), skiprows= 1, unpack =True)

###########################################################################
REF2_Dead = np.loadtxt("DeadStack/Ref/FilmStackRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_Dead   = np.loadtxt("DeadStack/Ref/FilmStackRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
###########################################################################
REFhBN_Vac = np.loadtxt("/scratch/dermoth/hBNWorking/Ref/hBN_VacRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
AhBN_Vac   = np.loadtxt("/scratch/dermoth/hBNWorking/Ref/hBN_VacRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
lamV       = np.loadtxt("/scratch/dermoth/hBNWorking/Ref/hBN_VacRTA.txt",  usecols=(0), skiprows= 1, unpack =True)

# ###########################################################################

REF2_no_hBN = np.loadtxt("../PitchSweep/3Term/2_30um/Ref/2_30umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_no_hBN   = np.loadtxt("../PitchSweep/3Term/2_30um/Ref/2_30umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
# ###########################################################################
# REF2_51um = np.loadtxt("2_51um/Ref/2_51umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
# A2_51um   = np.loadtxt("2_51um/Ref/2_51umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)

###########################################################################

# fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
# plt.setp(ax.spines.values(), linewidth=2)
# plt.tick_params(direction = 'in', width=2, labelsize=20)
# plt.ylabel("R", fontsize = '30')   
# plt.xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '30')
# plt.xlim(0,10)
# plt.ylim(0,1)

# plt.plot(lam, REF2_30um, label = r'$\rm R_{Pitch = \ 2.30 \ \mu m}$', color = "black", linewidth = 3)
# plt.plot(lam, TRA2_30um, label = r'$\rm T_{Pitch = \ 2.30 \ \mu m}$', color = "black", linewidth = 3)

# plt.plot(lam, REF2_281um, label = r'$\rm R_{Pitch = \ 2.281 \ \mu m}$', color = "red", linewidth = 3)
# plt.plot(lam, TRA2_281um, label = r'$\rm T_{Pitch = \ 2.281 \ \mu m}$', color = "red", linewidth = 3)

# plt.plot(lam, REF2_44um, label = r'$\rm R_{Pitch = \ 2.42 \ \mu m}$', color = "green", linewidth = 3)
# plt.plot(lam, TRA2_44um, label = r'$\rm T_{Pitch = \ 2.44 \ \mu m}$', color = "green", linewidth = 3))

# plt.plot(lam, REF2_30um, label = r'$\rm R_{Pitch = \ 2.30 \ \mu m}$', color = "orange", linewidth = 3)
# plt.plot(lam, TRA2_30um, label = r'$\rm T_{Pitch = \ 2.30 \ \mu m}$', color = "orange", linewidth = 3))

# plt.plot(lam, REF2_46um, label = r'$\rm R_{Pitch = \ 2.46 \ \mu m}$', color = "blue", linewidth = 3)

# plt.plot(lam, REF2_51um, label = r'$\rm R_{Pitch = \ 2.51 \ \mu m}$', color = "purple", linewidth = 3)

# plt.plot(lam, TRA2_46um, label = r'$\rm T_{Pitch = \ 2.46 \ \mu m}$', color = "blue", linewidth = 3))

# plt.plot(lam, REF2_48um, label = r'$\rm R_{Pitch = \ 2.48 \ \mu m}$', color = "purple", linewidth = 3)
# plt.plot(lam, TRA2_48um, label = r'$\rm T_{Pitch = \ 2.48 \ \mu m}$', color = "purple", linewidth = 3)
# plt.plot(lam, REFScan, label = r'$\rm R_{XFDTD \ dz = 1.0 \ nm}$', color = "green", linewidth = 3)
# plt.plot(lam, REF2um, label = r'$\rm R_{XFDTD \ dz = 2.0 \ nm}$', color = "blue", linewidth = 3)



# plt.plot(lam, TRA, label = r'$\rm T_{XFDTD}$', color = "black", linewidth = 4)
# plt.plot(lam, (1-(REF+TRA)), label = r'$\rm A_{XFDTD}$', color = "limegreen", linewidth = 2)


# plt.setp(ax.spines.values(), linewidth=2)
# plt.tick_params(left = False, bottom = False)   
# ax.legend(loc='center right', fontsize='22')
# # ax.axhline(y =1, color = 'black')
# # ax.axvline(x =0.1, color = 'black')
# # plt.savefig("XFAGFilmOnSiDrD3x3Sweep.pdf")
# plt.savefig("AgSiPitchSweepDDR_Fixed.png")
# plt.clf()

# fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
# plt.setp(ax.spines.values(), linewidth=2)
# plt.tick_params(direction = 'in', width=2, labelsize=20)
# plt.ylabel("R", fontsize = '30')   
# plt.xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '30')
# plt.xlim(5,9)
# plt.ylim(0,1)

# plt.plot(lam, REF2_30um, label = r'$\rm R_{Pitch = \ 2.30 \ \mu m}$', color = "black", linewidth = 3)
# # plt.plot(lam, TRA2_30um, label = r'$\rm T_{Pitch = \ 2.30 \ \mu m}$', color = "black", linewidth = 3))

# # plt.plot(lam, REF2_281, label = r'$\rm R_{Pitch = \ 2.281 \ \mu m}$', color = "red", linewidth = 3))
# # plt.plot(lam, TRA2_281, label = r'$\rm T_{Pitch = \ 2.281 \ \mu m}$', color = "red", linewidth = 3))

# # plt.plot(lam, REF2_30um, label = r'$\rm R_{Pitch = \ 2.30 \ \mu m}$', color = "orange", linewidth = 3)
# # plt.plot(lam, TRA2_30um, label = r'$\rm T_{Pitch = \ 2.30 \ \mu m}$', color = "orange", linewidth = 3))

# plt.plot(lam, REF2_42um, label = r'$\rm R_{Pitch = \ 2.42 \ \mu m}$', color = "green", linewidth = 3)
# # plt.plot(lam, TRA2_44um, label = r'$\rm T_{Pitch = \ 2.44 \ \mu m}$', color = "green", linewidth = 3))

# plt.plot(lam, REF2_46um, label = r'$\rm R_{Pitch = \ 2.46 \ \mu m}$', color = "blue", linewidth = 3)
# plt.plot(lam, TRA2_46um, label = r'$\rm T_{Pitch = \ 2.46 \ \mu m}$', color = "blue", linewidth = 3))
# plt.plot(lam, REF2_51um, label = r'$\rm R_{Pitch = \ 2.51 \ \mu m}$', color = "purple", linewidth = 3)


# plt.setp(ax.spines.values(), linewidth=2)
# plt.tick_params(left = False, bottom = False)   
# ax.legend(loc='center right', fontsize='22')
# # ax.axhline(y =1, color = 'black')
# # ax.axvline(x =0.1, color = 'black')
# # plt.savefig("XFAGFilmOnSiDrD3x3Sweep.pdf")
# plt.savefig("AgSiPitchSweepDDR5_9Fixed.png")

###############################################################################
for i in range(0,len(A2_30um)-1):
    if ((A2_30um[i] > A2_30um[i-1]) & (A2_30um[i] > A2_30um[i+1])):
    	peak  = lam[i]
    	peakA = A2_30um[i]
    	# print(lam[i])
for i in range(0,len(A2_30um)-1):
    if ((A2_30um[i] > A2_30um[i-1]) & (A2_30um[i] > A2_30um[i+1]) & (lam[i] < 7.5)):
    	peak2  = lam[i]
    	peakA2 = A2_30um[i]
###############################################################################

fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(direction = 'in', width=2, labelsize=20)
plt.ylabel("A", fontsize = '30')   
plt.xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '30')
plt.xlim(5,9)
plt.ylim(0,0.420)

plt.plot(lam, A2_30um, label = r'$\rm A_{Pitch = \ 2.30 \ \mu m}$', color = "black", linewidth = 3)

plt.plot(lam, A2_no_hBN, label = r'$\rm A_{Ag \ Pattern \ no \ hBN\ Pitch = \ 2.30 \ \mu m}$', color = "green", linewidth = 3)

plt.plot(lam, A2_Film, label = r'$\rm A_{hBN  \ Film \ Stack }$', color = "red", linewidth = 3)

plt.plot(lam, A2_Dead, label = r'$\rm A_{Dead \ Film \ Stack }$', color = "blue", linewidth = 3)

plt.plot(lamV, AhBN_Vac, label = r'$\rm A_{hBN \ Film \ in \ Vacuum}$', color = "fuchsia", linewidth = 3)

# plt.plot(lam, A2_50um, label = r'$\rm A_{Full \ Dev \ Pitch = \ 2.50 \ \mu m}$', color = "blue", linewidth = 3)

plt.scatter(peak, peakA,   linewidth = 3, s=55,edgecolors = 'black', c='red',  zorder = 25, label = r"$A_{ Plasmon   \ peak = %2.2f \ \mu m}$" %peak)       
plt.scatter(peak2, peakA2, linewidth = 3, s=55,edgecolors = 'black', c='blue', zorder = 25, label = r"$A_{ Polariton \ peak = %2.2f \ \mu m}$" %peak2)  
ax.axvline(x =peak, color = 'red')
ax.axvline(x =peak2, color = 'blue')

plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)   
ax.legend(loc='center left', fontsize='22')
# ax.axhline(y =1, color = 'black')
ax.axvline(x =7.26, color = 'black')
# plt.savefig("XFAGFilmOnSiDrD3x3Sweep.pdf")
plt.savefig("2_30umComp_A_5_9.png")


# plt.show()
