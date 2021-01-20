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
###########################  Pitch = 2.24 um ##############################
###########################################################################
lam        = np.loadtxt("2_24/Ref/2_24umPitchRTA.txt",  usecols=(0), skiprows= 1, unpack =True)
REF2_24um  = np.loadtxt("2_24/Ref/2_24umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_24um    = np.loadtxt("2_24/Ref/2_24umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)

###########################################################################
# REF2_281um = np.loadtxt("2_281um/Ref/2_281umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)

###########################################################################
# REF2_30um = np.loadtxt("2_30um/Ref/2_30umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
# A2_30um   = np.loadtxt("2_30um/Ref/2_30umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)

###########################################################################
REF2_36um = np.loadtxt("2_36um/Ref/2_36umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_36um   = np.loadtxt("2_36um/Ref/2_36umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)

###########################################################################
REF2_42um = np.loadtxt("2_42um/Ref/2_42umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_42um   = np.loadtxt("2_42um/Ref/2_42umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)

###########################################################################
REF2_48um = np.loadtxt("2_48um/Ref/2_48umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_48um   = np.loadtxt("2_48um/Ref/2_48umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)

###########################################################################
REF2_51um = np.loadtxt("2_51um/Ref/2_51umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_51um   = np.loadtxt("2_51um/Ref/2_51umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)

###########################################################################

fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(direction = 'in', width=2, labelsize=20)
plt.ylabel("R", fontsize = '30')   
plt.xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '30')
plt.xlim(0,10)
plt.ylim(0,1)

plt.plot(lam, REF2_24um, label = r'$\rm R_{Pitch = \ 2.24 \ \mu m}$', color = "black", linewidth = 3)
# plt.plot(lam, TRA2_24um, label = r'$\rm T_{Pitch = \ 2.24 \ \mu m}$', color = "black", linewidth = 3)

# plt.plot(lam, REF2_281um, label = r'$\rm R_{Pitch = \ 2.281 \ \mu m}$', color = "red", linewidth = 3)
# plt.plot(lam, TRA2_281um, label = r'$\rm T_{Pitch = \ 2.281 \ \mu m}$', color = "red", linewidth = 3)

plt.plot(lam, REF2_36um, label = r'$\rm R_{Pitch = \ 2.36 \ \mu m}$', color = "green", linewidth = 3)
# plt.plot(lam, TRA2_36um, label = r'$\rm T_{Pitch = \ 2.36 \ \mu m}$', color = "green", linewidth = 3))

# plt.plot(lam, REF2_30um, label = r'$\rm R_{Pitch = \ 2.30 \ \mu m}$', color = "orange", linewidth = 3)
# plt.plot(lam, TRA2_30um, label = r'$\rm T_{Pitch = \ 2.30 \ \mu m}$', color = "orange", linewidth = 3))

plt.plot(lam, REF2_42um, label = r'$\rm R_{Pitch = \ 2.42 \ \mu m}$', color = "blue", linewidth = 3)

plt.plot(lam, REF2_48um, label = r'$\rm R_{Pitch = \ 2.48 \ \mu m}$', color = "navy", linewidth = 3)


plt.plot(lam, REF2_51um, label = r'$\rm R_{Pitch = \ 2.51 \ \mu m}$', color = "purple", linewidth = 3)

# plt.plot(lam, TRA2_42um, label = r'$\rm T_{Pitch = \ 2.42 \ \mu m}$', color = "blue", linewidth = 3))

# plt.plot(lam, REF2_48um, label = r'$\rm R_{Pitch = \ 2.48 \ \mu m}$', color = "purple", linewidth = 3)
# plt.plot(lam, TRA2_48um, label = r'$\rm T_{Pitch = \ 2.48 \ \mu m}$', color = "purple", linewidth = 3)
# plt.plot(lam, REFScan, label = r'$\rm R_{XFDTD \ dz = 1.0 \ nm}$', color = "green", linewidth = 3)
# plt.plot(lam, REF2um, label = r'$\rm R_{XFDTD \ dz = 2.0 \ nm}$', color = "blue", linewidth = 3)



# plt.plot(lam, TRA, label = r'$\rm T_{XFDTD}$', color = "black", linewidth = 4)
# plt.plot(lam, (1-(REF+TRA)), label = r'$\rm A_{XFDTD}$', color = "limegreen", linewidth = 2)


plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)   
ax.legend(loc='center right', fontsize='22')
# ax.axhline(y =1, color = 'black')
# ax.axvline(x =0.1, color = 'black')
# plt.savefig("XFAGFilmOnSiDrD3x3Sweep.pdf")
plt.savefig("AgSiPitchSweepR.png")
plt.clf()

fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(direction = 'in', width=2, labelsize=20)
plt.ylabel("R", fontsize = '30')   
plt.xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '30')
plt.xlim(5,9)
plt.ylim(0,1)

plt.plot(lam, REF2_24um, label = r'$\rm R_{Pitch = \ 2.24 \ \mu m}$', color = "black", linewidth = 3)
# plt.plot(lam, TRA2_24um, label = r'$\rm T_{Pitch = \ 2.24 \ \mu m}$', color = "black", linewidth = 3))

# plt.plot(lam, REF2_281, label = r'$\rm R_{Pitch = \ 2.281 \ \mu m}$', color = "red", linewidth = 3))
# plt.plot(lam, TRA2_281, label = r'$\rm T_{Pitch = \ 2.281 \ \mu m}$', color = "red", linewidth = 3))

# plt.plot(lam, REF2_30um, label = r'$\rm R_{Pitch = \ 2.30 \ \mu m}$', color = "orange", linewidth = 3)
# plt.plot(lam, TRA2_30um, label = r'$\rm T_{Pitch = \ 2.30 \ \mu m}$', color = "orange", linewidth = 3))

plt.plot(lam, REF2_36um, label = r'$\rm R_{Pitch = \ 2.36 \ \mu m}$', color = "green", linewidth = 3)
# plt.plot(lam, TRA2_36um, label = r'$\rm T_{Pitch = \ 2.36 \ \mu m}$', color = "green", linewidth = 3))

plt.plot(lam, REF2_42um, label = r'$\rm R_{Pitch = \ 2.42 \ \mu m}$', color = "blue", linewidth = 3)

plt.plot(lam, REF2_48um, label = r'$\rm R_{Pitch = \ 2.48 \ \mu m}$', color = "navy", linewidth = 3)

# plt.plot(lam, TRA2_42um, label = r'$\rm T_{Pitch = \ 2.42 \ \mu m}$', color = "blue", linewidth = 3))
plt.plot(lam, REF2_51um, label = r'$\rm R_{Pitch = \ 2.51 \ \mu m}$', color = "purple", linewidth = 3)


plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)   
ax.legend(loc='center right', fontsize='22')
# ax.axhline(y =1, color = 'black')
# ax.axvline(x =0.1, color = 'black')
# plt.savefig("XFAGFilmOnSiDrD3x3Sweep.pdf")
plt.savefig("AgSiPitchSweepR5_9.png")

fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(direction = 'in', width=2, labelsize=20)
plt.ylabel("R", fontsize = '30')   
plt.xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '30')
plt.xlim(5,9)
plt.ylim(0,0.20)

plt.plot(lam, A2_24um, label = r'$\rm A_{Pitch = \ 2.24 \ \mu m}$', color = "black", linewidth = 3)

# plt.plot(lam, A2_281, label = r'$\rm A_{Pitch = \ 2.281 \ \mu m}$', color = "red", linewidth = 3))

# plt.plot(lam, A2_30um, label = r'$\rm A_{Pitch = \ 2.30 \ \mu m}$', color = "orange", linewidth = 3)

plt.plot(lam, A2_36um, label = r'$\rm A_{Pitch = \ 2.36 \ \mu m}$', color = "green", linewidth = 3)

plt.plot(lam, A2_42um, label = r'$\rm A_{Pitch = \ 2.42 \ \mu m}$', color = "blue", linewidth = 3)

plt.plot(lam, A2_48um, label = r'$\rm A_{Pitch = \ 2.48 \ \mu m}$', color = "navy", linewidth = 3)

plt.plot(lam, A2_51um, label = r'$\rm A_{Pitch = \ 2.51 \ \mu m}$', color = "purple", linewidth = 3)

plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)   
ax.legend(loc='center right', fontsize='22')
# ax.axhline(y =1, color = 'black')
ax.axvline(x =7.26, color = 'black')
# plt.savefig("XFAGFilmOnSiDrD3x3Sweep.pdf")
plt.savefig("AgSiPitchSweep_A_5_9.png")


# plt.show()
