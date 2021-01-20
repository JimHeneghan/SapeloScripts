import numpy as np
from scipy import *
from pylab import *
import math
import cmath
from ctypes import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.patches as mpatches

import matplotlib as mpl
AgRes = [1650.7100777 , 1612.3833622 , 1584.78655619, 1549.42671212,
         1515.61010203, 1487.20999405, 1459.85337525, 1433.48664951,
         1404.49339572, 1376.653309  , 1356.48325747, 1333.69005097,
         1308.55659926, 1293.32660286, 1269.68004063, 1249.68664107,
         1227.59666771, 1214.18001991, 1193.31671043, 1175.64003313]
ShadesRed  = ['deeppink','fuchsia', 'purple', 'darkviolet', 'blue', 'dodgerblue', 'deepskyblue', 'teal', 'springgreen', 'seagreen', 'limegreen',
 'forestgreen', 'greenyellow','gold', 'orange', 'orangered', 'salmon', 'red', 'darkred', 'lightcoral']
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

numPlots = 20
###########################################################################
lam        = np.loadtxt("2_38um/Ref/2_38umPitchRTA.txt",  usecols=(0), skiprows= 1, unpack =True)

###########################  d_{cc} = **** um ##############################
REF2_02um  = np.loadtxt("2_02um/Ref/2_02umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_02um    = np.loadtxt("2_02um/Ref/2_02umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
###########################  d_{cc} = **** um ##############################
REF2_06um  = np.loadtxt("2_06um/Ref/2_06umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_06um    = np.loadtxt("2_06um/Ref/2_06umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
###########################  d_{cc} = **** um ##############################
REF2_10um  = np.loadtxt("2_10um/Ref/2_10umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_10um    = np.loadtxt("2_10um/Ref/2_10umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)

###########################  d_{cc} = **** um ##############################
REF2_14um  = np.loadtxt("2_14um/Ref/2_14umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_14um    = np.loadtxt("2_14um/Ref/2_14umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)

###########################  d_{cc} = **** um ##############################
REF2_18um  = np.loadtxt("2_18um/Ref/2_18umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_18um    = np.loadtxt("2_18um/Ref/2_18umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
###########################  d_{cc} = **** um ##############################
REF2_22um  = np.loadtxt("2_22um/Ref/2_22umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_22um    = np.loadtxt("2_22um/Ref/2_22umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
###########################  d_{cc} = **** um ##############################
REF2_26um  = np.loadtxt("2_26um/Ref/2_26umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_26um    = np.loadtxt("2_26um/Ref/2_26umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
###########################################################################
REF2_30um  = np.loadtxt("2_30um/Ref/2_30umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_30um    = np.loadtxt("2_30um/Ref/2_30umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)

###########################################################################
REF2_34um  = np.loadtxt("2_34um/Ref/2_34umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_34um    = np.loadtxt("2_34um/Ref/2_34umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)

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
REF2_58um = np.loadtxt("2_58um/Ref/2_58umPitchRTA.txt",    usecols=(1), skiprows= 1, unpack =True)
A2_58um   = np.loadtxt("2_58um/Ref/2_58umPitchRTA.txt",    usecols=(3), skiprows= 1, unpack =True)
###########################################################################
REF2_62um = np.loadtxt("2_62um/Ref/2_62umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_62um   = np.loadtxt("2_62um/Ref/2_62umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
###########################################################################

REF2_66um = np.loadtxt("2_66um/Ref/2_66umPitchRTA.txt",    usecols=(1), skiprows= 1, unpack =True)
A2_66um   = np.loadtxt("2_66um/Ref/2_66umPitchRTA.txt",    usecols=(3), skiprows= 1, unpack =True)
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

peak  = np.zeros(numPlots, dtype = np.double)
peakA = np.zeros(numPlots, dtype = np.double)

peak2  = np.zeros(numPlots, dtype = np.double)
peakA2 = np.zeros(numPlots, dtype = np.double)
pitchThou = np.zeros((numPlots,NFREQs), dtype = np.double)
pitchThou[0]  = REF2_02um
pitchThou[1]  = REF2_06um
pitchThou[2]  = REF2_10um
pitchThou[3]  = REF2_14um
pitchThou[4]  = REF2_18um
pitchThou[5]  = REF2_22um
pitchThou[6]  = REF2_26um
pitchThou[7]  = REF2_30um
pitchThou[8]  = REF2_34um
pitchThou[9]  = REF2_38um
pitchThou[10]  = REF2_42um
pitchThou[11] = REF2_46um
pitchThou[12] = REF2_50um
pitchThou[13] = REF2_54um
pitchThou[14] = REF2_58um
pitchThou[15] = REF2_62um
pitchThou[16] = REF2_66um
pitchThou[17] = REF2_70um
pitchThou[18] = REF2_74um
pitchThou[19] = REF2_78um
###############################################################################

fig, (ax1, ax2, ax3) = plt.subplots(3, figsize = (18,27), sharex = True)
# plt.rc('axes', linewidth=4) 
fig.subplots_adjust(hspace=0)
plt.setp(ax1.spines.values(), linewidth=4)



ax1.tick_params(direction = 'in', width=2, labelsize=20)
ax1.set_ylabel("Reflectance", fontsize = '40')   
# ax1.xlabel(r"$\rm Wavenumber \ (cm^{-1})$", fontsize = '30')
ax1.set_xlim(1000,2000)
ax1.set_ylim(0.38,0.9)


ax1.plot(1/(lam*1e-4), REF2_02um, label = r'$\rm R_{d_{cc}\ 2.02 \ \mu m}$', color = ShadesRed[0], linewidth = 3)

ax1.plot(1/(lam*1e-4), REF2_06um, label = r'$\rm R_{d_{cc}\ 2.06 \ \mu m}$', color = ShadesRed[1], linewidth = 3)

ax1.plot(1/(lam*1e-4), REF2_10um, label = r'$\rm R_{d_{cc}\ 2.10 \ \mu m}$', color = ShadesRed[2], linewidth = 3)

ax1.plot(1/(lam*1e-4), REF2_14um, label = r'$\rm R_{d_{cc}\ 2.14 \ \mu m}$', color = ShadesRed[3], linewidth = 3)

ax1.plot(1/(lam*1e-4), REF2_18um, label = r'$\rm R_{d_{cc}\ 2.18 \ \mu m}$', color = ShadesRed[4], linewidth = 3)

ax1.plot(1/(lam*1e-4), REF2_22um, label = r'$\rm R_{d_{cc}\ 2.22 \ \mu m}$', color = ShadesRed[5], linewidth = 3)

ax1.plot(1/(lam*1e-4), REF2_26um, label = r'$\rm R_{d_{cc}\ 2.26 \ \mu m}$', color = ShadesRed[6], linewidth = 3)

ax1.plot(1/(lam*1e-4), REF2_30um, label = r'$\rm R_{d_{cc}\ 2.30 \ \mu m}$', color = ShadesRed[7], linewidth = 3)

ax1.plot(1/(lam*1e-4), REF2_34um, label = r'$\rm R_{d_{cc}\ 2.34 \ \mu m}$', color = ShadesRed[8], linewidth = 3)

ax1.plot(1/(lam*1e-4), REF2_38um, label = r'$\rm R_{d_{cc}\ 2.38 \ \mu m}$', color = ShadesRed[9], linewidth = 3)

ax1.plot(1/(lam*1e-4), REF2_42um, label = r'$\rm R_{d_{cc}\ 2.42 \ \mu m}$', color = ShadesRed[10], linewidth = 3)

ax1.plot(1/(lam*1e-4), REF2_46um, label = r'$\rm R_{d_{cc}\ 2.46 \ \mu m}$', color = ShadesRed[11], linewidth = 3)

ax1.plot(1/(lam*1e-4), REF2_50um, label = r'$\rm R_{d_{cc}\ 2.50 \ \mu m}$', color = ShadesRed[12], linewidth = 3)

ax1.plot(1/(lam*1e-4), REF2_54um, label = r'$\rm R_{d_{cc}\ 2.54 \ \mu m}$', color = ShadesRed[13], linewidth = 3)

ax1.plot(1/(lam*1e-4), REF2_58um, label = r'$\rm R_{d_{cc}\ 2.58 \ \mu m}$', color = ShadesRed[14], linewidth = 3)

ax1.plot(1/(lam*1e-4), REF2_62um, label = r'$\rm R_{d_{cc}\ 2.62 \ \mu m}$', color = ShadesRed[15], linewidth = 3)

ax1.plot(1/(lam*1e-4), REF2_66um, label = r'$\rm R_{d_{cc}\ 2.66 \ \mu m}$', color = ShadesRed[16], linewidth = 3)

ax1.plot(1/(lam*1e-4), REF2_70um, label = r'$\rm R_{d_{cc}\ 2.70 \ \mu m}$', color = ShadesRed[17], linewidth = 3)

ax1.plot(1/(lam*1e-4), REF2_74um, label = r'$\rm R_{d_{cc}\ 2.74 \ \mu m}$', color = ShadesRed[18], linewidth = 3)

ax1.plot(1/(lam*1e-4), REF2_78um, label = r'$\rm R_{d_{cc}\ 2.78 \ \mu m}$', color = ShadesRed[19], linewidth = 3)


ax1.tick_params(direction = 'in', width=2, labelsize=25)   
plt.setp(ax1.spines.values(), linewidth=4)

#######################################################################

plt.setp(ax2.spines.values(), linewidth=4)

ax2.tick_params(direction = 'in', width=2, labelsize=25)
ax2.set_ylabel("Absorptance", fontsize = '40')   
# ax2.xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '30')
ax2.set_xlim(1000,2000)
ax2.set_ylim(0,0.30)

ax2.plot((1/(lam*1e-4)), A2_02um, label = r'$\rm A_{d_{cc}\ 2.02 \ \mu m}$', color = ShadesRed[0], linewidth = 3)

ax2.plot((1/(lam*1e-4)), A2_06um, label = r'$\rm A_{d_{cc}\ 2.06 \ \mu m}$', color = ShadesRed[1], linewidth = 3)

ax2.plot((1/(lam*1e-4)), A2_10um, label = r'$\rm A_{d_{cc}\ 2.10 \ \mu m}$', color = ShadesRed[2], linewidth = 3)

ax2.plot((1/(lam*1e-4)), A2_14um, label = r'$\rm A_{d_{cc}\ 2.14 \ \mu m}$', color = ShadesRed[3], linewidth = 3)

ax2.plot((1/(lam*1e-4)), A2_18um, label = r'$\rm A_{d_{cc}\ 2.18 \ \mu m}$', color = ShadesRed[4], linewidth = 3)

ax2.plot((1/(lam*1e-4)), A2_22um, label = r'$\rm A_{d_{cc}\ 2.22 \ \mu m}$', color = ShadesRed[5], linewidth = 3)

ax2.plot((1/(lam*1e-4)), A2_26um, label = r'$\rm A_{d_{cc}\ 2.26 \ \mu m}$', color = ShadesRed[6], linewidth = 3)

ax2.plot((1/(lam*1e-4)), A2_30um, label = r'$\rm A_{d_{cc}\ 2.30 \ \mu m}$', color = ShadesRed[7], linewidth = 6)

ax2.plot((1/(lam*1e-4)), A2_34um, label = r'$\rm A_{d_{cc}\ 2.34 \ \mu m}$', color = ShadesRed[8], linewidth = 6)

ax2.plot((1/(lam*1e-4)), A2_38um, label = r'$\rm A_{d_{cc}\ 2.38 \ \mu m}$', color = ShadesRed[9], linewidth = 3)

ax2.plot((1/(lam*1e-4)), A2_42um, label = r'$\rm A_{d_{cc}\ 2.42 \ \mu m}$', color = ShadesRed[10], linewidth = 3)

ax2.plot((1/(lam*1e-4)), A2_46um, label = r'$\rm A_{d_{cc}\ 2.46 \ \mu m}$', color = ShadesRed[11], linewidth = 3)

ax2.plot((1/(lam*1e-4)), A2_50um, label = r'$\rm A_{d_{cc}\ 2.50 \ \mu m}$', color = ShadesRed[12], linewidth = 3)

ax2.plot((1/(lam*1e-4)), A2_54um, label = r'$\rm A_{d_{cc}\ 2.54 \ \mu m}$', color = ShadesRed[13], linewidth = 3)

ax2.plot((1/(lam*1e-4)), A2_58um, label = r'$\rm A_{d_{cc}\ 2.58 \ \mu m}$', color = ShadesRed[14], linewidth = 3)

ax2.plot((1/(lam*1e-4)), A2_62um, label = r'$\rm A_{d_{cc}\ 2.62 \ \mu m}$', color = ShadesRed[15], linewidth = 3)

ax2.plot((1/(lam*1e-4)), A2_66um, label = r'$\rm A_{d_{cc}\ 2.66 \ \mu m}$', color = ShadesRed[16], linewidth = 3)

ax2.plot((1/(lam*1e-4)), A2_70um, label = r'$\rm A_{d_{cc}\ 2.70 \ \mu m}$', color = ShadesRed[17], linewidth = 3)

ax2.plot((1/(lam*1e-4)), A2_74um, label = r'$\rm A_{d_{cc}\ 2.74 \ \mu m}$', color = ShadesRed[18], linewidth = 3)

ax2.plot((1/(lam*1e-4)), A2_78um, label = r'$\rm A_{d_{cc}\ 2.78 \ \mu m}$', color = ShadesRed[19], linewidth = 3)

plt.setp(ax2.spines.values(), linewidth=4)

###############################################################################
###############################################################################



###########################  d_{cc} = **** um ##############################
TRA2_02um  = np.loadtxt("2_02um/Ref/2_02umPitchRTA.txt",  usecols=(2), skiprows= 1, unpack =True)
###########################  d_{cc} = **** um ##############################
TRA2_06um  = np.loadtxt("2_06um/Ref/2_06umPitchRTA.txt",  usecols=(2), skiprows= 1, unpack =True)
###########################  d_{cc} = **** um ##############################
TRA2_10um  = np.loadtxt("2_10um/Ref/2_10umPitchRTA.txt",  usecols=(2), skiprows= 1, unpack =True)

###########################  d_{cc} = **** um ##############################
TRA2_14um  = np.loadtxt("2_14um/Ref/2_14umPitchRTA.txt",  usecols=(2), skiprows= 1, unpack =True)

###########################  d_{cc} = **** um ##############################
TRA2_18um  = np.loadtxt("2_18um/Ref/2_18umPitchRTA.txt",  usecols=(2), skiprows= 1, unpack =True)
###########################  d_{cc} = **** um ##############################
TRA2_22um  = np.loadtxt("2_22um/Ref/2_22umPitchRTA.txt",  usecols=(2), skiprows= 1, unpack =True)
###########################  d_{cc} = **** um ##############################
TRA2_26um  = np.loadtxt("2_26um/Ref/2_26umPitchRTA.txt",  usecols=(2), skiprows= 1, unpack =True)
###########################################################################
TRA2_30um  = np.loadtxt("2_30um/Ref/2_30umPitchRTA.txt",  usecols=(2), skiprows= 1, unpack =True)

###########################################################################
TRA2_34um  = np.loadtxt("2_34um/Ref/2_34umPitchRTA.txt",  usecols=(2), skiprows= 1, unpack =True)

###########################################################################
TRA2_38um = np.loadtxt("2_38um/Ref/2_38umPitchRTA.txt",  usecols=(2), skiprows= 1, unpack =True)

###########################################################################
TRA2_42um = np.loadtxt("2_42um/Ref/2_42umPitchRTA.txt",  usecols=(2), skiprows= 1, unpack =True)

###########################################################################
TRA2_46um = np.loadtxt("2_46um/Ref/2_46umPitchRTA.txt",  usecols=(2), skiprows= 1, unpack =True)
###########################################################################

TRA2_50um = np.loadtxt("2_50um/Ref/2_50umPitchRTA.txt",  usecols=(2), skiprows= 1, unpack =True)
###########################################################################
TRA2_54um = np.loadtxt("2_54um/Ref/2_54umPitchRTA.txt",  usecols=(2), skiprows= 1, unpack =True)
###########################################################################
TRA2_58um = np.loadtxt("2_58um/Ref/2_58umPitchRTA.txt",    usecols=(2), skiprows= 1, unpack =True)
###########################################################################
TRA2_62um = np.loadtxt("2_62um/Ref/2_62umPitchRTA.txt",  usecols=(2), skiprows= 1, unpack =True)
# TRA2_62um = np.loadtxt("2_66um/Ref/2_66umPitchRTA.txt",    usecols=(2), skiprows= 1, unpack =True)

###########################################################################

TRA2_66um = np.loadtxt("2_66um/Ref/2_66umPitchRTA.txt",    usecols=(2), skiprows= 1, unpack =True)
###########################################################################
TRA2_70um = np.loadtxt("2_70um/Ref/2_70umPitchRTA.txt",  usecols=(2), skiprows= 1, unpack =True)

###########################################################################
TRA2_74um = np.loadtxt("2_74um/Ref/2_74umPitchRTA.txt",  usecols=(2), skiprows= 1, unpack =True)

###########################################################################
TRA2_78um = np.loadtxt("2_78um/Ref/2_78umPitchRTA.txt",  usecols=(2), skiprows= 1, unpack =True)
###########################################################################
TraTroughLam   = np.zeros(numPlots, dtype = np.double)
TraTrough      = np.zeros(numPlots, dtype = np.double)

TraTroughLam2  = np.zeros(numPlots, dtype = np.double)
TraTrough2     = np.zeros(numPlots, dtype = np.double)

pitchThou = np.zeros((numPlots,NFREQs), dtype = np.double)


pitchThou[0]   = TRA2_02um
pitchThou[1]   = TRA2_06um
pitchThou[2]   = TRA2_10um
pitchThou[3]   = TRA2_14um
pitchThou[4]   = TRA2_18um
pitchThou[5]   = TRA2_22um
pitchThou[6]   = TRA2_26um
pitchThou[7]   = TRA2_30um
pitchThou[8]   = TRA2_34um
pitchThou[9]   = TRA2_38um
pitchThou[10]  = TRA2_42um
pitchThou[11]  = TRA2_46um
pitchThou[12]  = TRA2_50um
pitchThou[13]  = TRA2_54um
pitchThou[14]  = TRA2_58um
pitchThou[15]  = TRA2_62um
pitchThou[16]  = TRA2_66um
pitchThou[17]  = TRA2_70um
pitchThou[18]  = TRA2_74um
pitchThou[19]  = TRA2_78um
###############################################################################
plt.setp(ax3.spines.values(), linewidth=4)

ax3.tick_params(direction = 'in', width=2, labelsize=25)
ax3.set_ylabel("Transmittance", fontsize = '40')   
ax3.set_xlabel(r"$\rm Frequency \ (cm^{-1})$", fontsize = '50')
ax3.set_xlim(1000,2000)
ax3.set_ylim(0,0.42)

ax3.plot(1/(lam*1e-4), TRA2_02um, label = r'$\rm T_{d_{cc}\ 2.02 \ \mu m}$', color = ShadesRed[0], linewidth = 3)

ax3.plot(1/(lam*1e-4), TRA2_06um, label = r'$\rm T_{d_{cc}\ 2.06 \ \mu m}$', color = ShadesRed[1], linewidth = 3)

ax3.plot(1/(lam*1e-4), TRA2_10um, label = r'$\rm T_{d_{cc}\ 2.10 \ \mu m}$', color = ShadesRed[2], linewidth = 3)

ax3.plot(1/(lam*1e-4), TRA2_14um, label = r'$\rm T_{d_{cc}\ 2.14 \ \mu m}$', color = ShadesRed[3], linewidth = 3)

ax3.plot(1/(lam*1e-4), TRA2_18um, label = r'$\rm T_{d_{cc}\ 2.18 \ \mu m}$', color = ShadesRed[4], linewidth = 3)

ax3.plot(1/(lam*1e-4), TRA2_22um, label = r'$\rm T_{d_{cc}\ 2.22 \ \mu m}$', color = ShadesRed[5], linewidth = 3)

ax3.plot(1/(lam*1e-4), TRA2_26um, label = r'$\rm T_{d_{cc}\ 2.26 \ \mu m}$', color = ShadesRed[6], linewidth = 3)

ax3.plot(1/(lam*1e-4), TRA2_30um, label = r'$\rm T_{d_{cc}\ 2.30 \ \mu m}$', color = ShadesRed[7], linewidth = 3)

ax3.plot(1/(lam*1e-4), TRA2_34um, label = r'$\rm T_{d_{cc}\ 2.34 \ \mu m}$', color = ShadesRed[8], linewidth = 3)

ax3.plot(1/(lam*1e-4), TRA2_38um, label = r'$\rm T_{d_{cc}\ 2.38 \ \mu m}$', color = ShadesRed[9], linewidth = 3)

ax3.plot(1/(lam*1e-4), TRA2_42um, label = r'$\rm T_{d_{cc}\ 2.42 \ \mu m}$', color = ShadesRed[10], linewidth = 3)

ax3.plot(1/(lam*1e-4), TRA2_46um, label = r'$\rm T_{d_{cc}\ 2.46 \ \mu m}$', color = ShadesRed[11], linewidth = 3)

ax3.plot(1/(lam*1e-4), TRA2_50um, label = r'$\rm T_{d_{cc}\ 2.50 \ \mu m}$', color = ShadesRed[12], linewidth = 3)

ax3.plot(1/(lam*1e-4), TRA2_54um, label = r'$\rm T_{d_{cc}\ 2.54 \ \mu m}$', color = ShadesRed[13], linewidth = 3)

ax3.plot(1/(lam*1e-4), TRA2_58um, label = r'$\rm T_{d_{cc}\ 2.58 \ \mu m}$', color = ShadesRed[14], linewidth = 3)

ax3.plot(1/(lam*1e-4), TRA2_62um, label = r'$\rm T_{d_{cc}\ 2.62 \ \mu m}$', color = ShadesRed[15], linewidth = 3)

ax3.plot(1/(lam*1e-4), TRA2_66um, label = r'$\rm T_{d_{cc}\ 2.66 \ \mu m}$', color = ShadesRed[16], linewidth = 3)

ax3.plot(1/(lam*1e-4), TRA2_70um, label = r'$\rm T_{d_{cc}\ 2.70 \ \mu m}$', color = ShadesRed[17], linewidth = 3)

ax3.plot(1/(lam*1e-4), TRA2_74um, label = r'$\rm T_{d_{cc}\ 2.74 \ \mu m}$', color = ShadesRed[18], linewidth = 3)

ax3.plot(1/(lam*1e-4), TRA2_78um, label = r'$\rm T_{d_{cc}\ 2.78 \ \mu m}$', color = ShadesRed[19], linewidth = 3)



###############################################################################
patches = []
PitchLeng = np.linspace(2.02, 2.78, 20)
dcc = np.zeros(numPlots, dtype = np.double)
dcc = PitchLeng

for i in range(0, 20):
    temp = mpatches.Patch(facecolor=ShadesRed[i], label = r'$\rm d_{cc} = %2.2f \ \mu m$' %dcc[i], edgecolor='black')
    patches.append(temp) 
leg = ax1.legend(handles = patches, ncol = 4, loc = 'upper center', frameon = True,fancybox = False, fontsize = 25, bbox_to_anchor=(0., 1.35, 1., .102),mode="expand", borderaxespad=0.) #bbox_to_anchor=(1.05, 0.5),
leg.get_frame().set_edgecolor('black')

leg.get_frame().set_linewidth(4)
###############################################################################  


plt.savefig("BareRTA_Sweep.png")



###############################################################################
