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
###########################  Pitch = 2.44 um ##############################
###########################################################################

lam       = np.loadtxt("2_44umPitchRTA.txt",  usecols=(0), skiprows= 1, unpack =True)
REF2_44um = np.loadtxt("2_44umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
TRA2_44um = np.loadtxt("2_44umPitchRTA.txt",  usecols=(2), skiprows= 1, unpack =True)
A2_44um   = np.loadtxt("2_44umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)


###########################################################################

fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(direction = 'in', width=2, labelsize=20)
plt.ylabel("R", fontsize = '30')   
plt.xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '30')
plt.xlim(0,10)
plt.ylim(0,1)



plt.plot(lam, REF2_44um, label = r'$\rm R_{Pitch = \ 2.42 \ \mu m}$', color = "red", linewidth = 3)
plt.plot(lam, TRA2_44um, label = r'$\rm T_{Pitch = \ 2.42 \ \mu m}$', color = "black", linewidth = 3)
plt.plot(lam, REF2_44um + TRA2_44um, label = r'$\rm R+T_{Pitch = \ 2.42 \ \mu m}$', color = "limegreen", linewidth = 3)
plt.plot(lam,   A2_44um, label = r'$\rm A_{Pitch = \ 2.42 \ \mu m}$', color = "cyan", linewidth = 3)

plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)   
ax.legend(loc='center right', fontsize='22')
# ax.axhline(y =1, color = 'black')
# ax.axvline(x =0.1, color = 'black')
# plt.savefig("XFAGFilmOnSiDrD3x3Sweep.pdf")
plt.savefig("FullDev2_44RTA.png")
plt.clf()

fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(direction = 'in', width=2, labelsize=20)
plt.ylabel("R", fontsize = '30')   
plt.xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '30')
plt.xlim(5,9)
plt.ylim(0,1)

plt.plot(lam, REF2_44um, label = r'$\rm R_{Pitch = \ 2.42 \ \mu m}$', color = "red", linewidth = 3)
plt.plot(lam, TRA2_44um, label = r'$\rm T_{Pitch = \ 2.42 \ \mu m}$', color = "black", linewidth = 3)
plt.plot(lam, REF2_44um + TRA2_44um, label = r'$\rm R+T_{Pitch = \ 2.42 \ \mu m}$', color = "limegreen", linewidth = 3)
plt.plot(lam,   A2_44um, label = r'$\rm A_{Pitch = \ 2.42 \ \mu m}$', color = "cyan", linewidth = 3)

plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)   
ax.legend(loc='center right', fontsize='22')
# ax.axhline(y =1, color = 'black')
# ax.axvline(x =0.1, color = 'black')
# plt.savefig("XFAGFilmOnSiDrD3x3Sweep.pdf")
plt.savefig("FullDev2_44RTA5_9.png")

# fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
# plt.setp(ax.spines.values(), linewidth=2)
# plt.tick_params(direction = 'in', width=2, labelsize=20)
# plt.ylabel("R", fontsize = '30')   
# plt.xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '30')
# plt.xlim(5,9)
# plt.ylim(0,0.30)

# plt.plot(lam, A2_30um, label = r'$\rm A_{Pitch = \ 2.30 \ \mu m}$', color = "black", linewidth = 3)

# # plt.plot(lam, A2_281, label = r'$\rm A_{Pitch = \ 2.281 \ \mu m}$', color = "red", linewidth = 3))

# plt.plot(lam, A2_38um, label = r'$\rm A_{Pitch = \ 2.38 \ \mu m}$', color = "red", linewidth = 3)

# # plt.plot(lam, A2_43um, label = r'$\rm A_{Pitch = \ 2.43 \ \mu m}$', color = "green", linewidth = 3)

# plt.plot(lam, A2_44um, label = r'$\rm A_{Pitch = \ 2.44 \ \mu m}$', color = "blue", linewidth = 3)

# plt.plot(lam, A2_46um, label = r'$\rm A_{Pitch = \ 2.46 \ \mu m}$', color = "black", linewidth = 3)

# # plt.plot(lam, A2_51um, label = r'$\rm A_{Pitch = \ 2.51 \ \mu m}$', color = "purple", linewidth = 3)

# plt.setp(ax.spines.values(), linewidth=2)
# plt.tick_params(left = False, bottom = False)   
# ax.legend(loc='center left', fontsize='22')
# # ax.axhline(y =1, color = 'black')
# ax.axvline(x =7.26, color = 'black')
# # plt.savefig("XFAGFilmOnSiDrD3x3Sweep.pdf")
# plt.savefig("AgSiPitchSweeDDp_A_5_9.png")


# # plt.show()
