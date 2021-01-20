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

# res = [6.598003, 6.724, 6.885993, 7.029992, 7.173996, 7.299998, 7.426, 7.570004, 7.677995, 7.821992]
# res = [5.94,  6.21,6.35, 6.48, 6.62, 6.75, 6.89, 7.03, 7.16, 7.30, 7.43, 7.57, 7.71, 7.84,  8.11]
res = [1650.7100777 , 1612.3833622 , 1584.78655619, 1549.42671212,
       1515.61010203, 1487.20999405, 1459.85337525, 1433.48664951,
       1404.49339572, 1376.653309  , 1356.48325747, 1333.69005097,
       1308.55659926, 1293.32660286, 1269.68004063, 1249.68664107,
       1227.59666771, 1214.18001991, 1193.31671043, 1155.64003313, 1125] #1193.31671043,
res = np.asarray(res)
Dots = [1650.7100777 , 1612.3833622 , 1584.78655619, 1549.42671212,
       1515.61010203, 1487.20999405, 1459.85337525, 1433.48664951,
       1404.49339572, 1376.653309  , 1356.48325747, 1333.69005097,
       1308.55659926, 1293.32660286, 1269.68004063, 1249.68664107,
       1227.59666771, 1214.18001991, 1193.31671043, 1155.64003313,]
##############################################################################

##############################################################################
##############################################################################

dx     = 20e-9
dy     = 20e-9
c0     = 3e8
nref   = 1.0
ntra   = 3.42
NFREQs = 500

Full = []#np.zeros((NFREQs,10), dtype = np.double)
###########################################################################
lam        = np.loadtxt("2_38um/Ref/2_38umPitchRTA.txt",  usecols=(0), skiprows= 1, unpack =True)
lam = np.append(lam, 10.018)
###########################  Pitch = **** um ##############################
REF2_02um  = np.loadtxt("2_02um/Ref/2_02umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_02um    = np.loadtxt("2_02um/Ref/2_02umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
Full.append(REF2_02um)
###########################  Pitch = **** um ##############################
REF2_06um  = np.loadtxt("2_06um/Ref/2_06umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_06um    = np.loadtxt("2_06um/Ref/2_06umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
Full.append(REF2_06um)
###########################  Pitch = **** um ##############################
REF2_10um  = np.loadtxt("2_10um/Ref/2_10umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_10um    = np.loadtxt("2_10um/Ref/2_10umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
Full.append(REF2_10um)

###########################  Pitch = **** um ##############################
REF2_14um  = np.loadtxt("2_14um/Ref/2_14umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_14um    = np.loadtxt("2_14um/Ref/2_14umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
Full.append(REF2_14um)

###########################  Pitch = **** um ##############################
REF2_18um  = np.loadtxt("2_18um/Ref/2_18umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_18um    = np.loadtxt("2_18um/Ref/2_18umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
Full.append(REF2_18um)

###########################  Pitch = **** um ##############################
REF2_22um  = np.loadtxt("2_22um/Ref/2_26umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_22um    = np.loadtxt("2_22um/Ref/2_26umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
Full.append(REF2_22um)
###########################  Pitch = **** um ##############################
REF2_26um  = np.loadtxt("2_26um/Ref/2_26umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_26um    = np.loadtxt("2_26um/Ref/2_26umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
# Full[,1]    = REF2_26um
Full.append(REF2_26um)

###########################################################################
REF2_30um  = np.loadtxt("2_30um/Ref/2_30umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_30um    = np.loadtxt("2_30um/Ref/2_30umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
# Full[,2]    = REF2_30um
Full.append(REF2_30um)

###########################################################################
REF2_34um  = np.loadtxt("2_34um/Ref/2_34umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_34um    = np.loadtxt("2_34um/Ref/2_34umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
Full.append(REF2_34um)

# Full[,3]    = REF2_34um
###########################################################################
REF2_38um = np.loadtxt("2_38um/Ref/2_38umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_38um   = np.loadtxt("2_38um/Ref/2_38umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
Full.append(REF2_38um)

# Full[4]    = REF2_38um
###########################################################################
REF2_42um = np.loadtxt("2_42um/Ref/2_42umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_42um   = np.loadtxt("2_42um/Ref/2_42umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
Full.append(REF2_42um)

# Full[5]    = REF2_42um
###########################################################################
REF2_46um = np.loadtxt("2_46um/Ref/2_46umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_46um   = np.loadtxt("2_46um/Ref/2_46umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
Full.append(REF2_46um)
# Full[6]    = REF2_46um
###########################################################################
REF2_50um = np.loadtxt("2_50um/Ref/2_50umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_50um   = np.loadtxt("2_50um/Ref/2_50umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
Full.append(REF2_50um)
# Full[7]    = REF2_50um
###########################################################################
REF2_54um = np.loadtxt("2_54um/Ref/2_50umPitchRTA.txt",  usecols=(1), skiprows= 1, unpack =True)
A2_54um   = np.loadtxt("2_54um/Ref/2_50umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
Full.append(REF2_54um)
# Full[8]    = REF2_54um
###########################################################################
REF2_58um = np.loadtxt("2_58um/Ref/2_58umPitchRTA.txt",    usecols=(1), skiprows= 1, unpack =True)
A2_58um   = np.loadtxt("2_58um/Ref/2_58umPitchRTA.txt",    usecols=(3), skiprows= 1, unpack =True)
Full.append(REF2_58um)
###########################################################################
REF2_62um = np.loadtxt("2_62um/Ref/2_62umPitchRTA.txt",    usecols=(1), skiprows= 1, unpack =True)
A2_62um   = np.loadtxt("2_62um/Ref/2_62umPitchRTA.txt",    usecols=(3), skiprows= 1, unpack =True)
Full.append(REF2_62um)
###########################################################################
REF2_66um = np.loadtxt("2_66um/Ref/2_66umPitchRTA.txt",    usecols=(1), skiprows= 1, unpack =True)
A2_66um   = np.loadtxt("2_66um/Ref/2_66umPitchRTA.txt",    usecols=(3), skiprows= 1, unpack =True)
Full.append(REF2_66um)
###########################################################################
REF2_70um = np.loadtxt("2_70um/Ref/2_70umPitchRTA.txt",    usecols=(1), skiprows= 1, unpack =True)
A2_70um   = np.loadtxt("2_70um/Ref/2_70umPitchRTA.txt",    usecols=(3), skiprows= 1, unpack =True)
Full.append(REF2_70um)
###########################################################################
REF2_74um = np.loadtxt("2_74um/Ref/2_74umPitchRTA.txt",    usecols=(1), skiprows= 1, unpack =True)
A2_74um   = np.loadtxt("2_74um/Ref/2_74umPitchRTA.txt",    usecols=(3), skiprows= 1, unpack =True)
Full.append(REF2_74um)
###########################################################################
REF2_78um = np.loadtxt("2_78um/Ref/2_78umPitchRTA.txt",    usecols=(1), skiprows= 1, unpack =True)
A2_78um   = np.loadtxt("2_78um/Ref/2_78umPitchRTA.txt",    usecols=(3), skiprows= 1, unpack =True)
Full.append(REF2_78um)
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
for i in range(0, numPlots):
    for j in range(0,NFREQs-1):
        if ((pitchThou[i,j] < pitchThou[i,j-1]) & (pitchThou[i,j+1] < pitchThou[i,j]) & (lam[j] > 5.0)):
            peak[i]  = lam[j]
            # print(j)
            peakA[i] = pitchThou[i,j]
# temp = 1.0
for i in range(0, numPlots):
    temp = 1.0
    for q in range(0,299):
        j = NFREQs - 2 - q
        if (((pitchThou[i,j] < pitchThou[i,j-1]) & (pitchThou[i,j] < pitchThou[i,j+1])) & (pitchThou[i,j] < temp) & (lam[j] < peak[i]) & (lam[j] > 5.0)):
            temp = pitchThou[i,j]
            peak2[i]  = lam[j]
            # print(j)
            peakA2[i] = pitchThou[i,j]
###############################################################################

peak = np.asarray(peak)

PitchLeng = np.linspace(2.02, 2.78, 20)
dcc = np.zeros(numPlots, dtype = np.double)
dcc = PitchLeng
dcc1 = dcc
###############################################################################

# X = (c0/(lam*1e-6))*1e-12
# Y = (c0/(res*1e-6))*1e-12
# minni = (c0/(10.0*1e-6))*1e-12
# maxi  = (c0/(4.5*1e-6))*1e-12
# # Full = np.reshape(Full, (10,NFREQs), order='C')
Full = np.transpose(Full)
# fig, ax = plt.subplots(figsize=(9,12),constrained_layout=True)
# plt.setp(ax.spines.values(), linewidth=2)
# plt.tick_params(direction = 'in', width=2, labelsize=20)
# plt.ylabel(r"$\rm Frequency \ (THz)$", fontsize = '30')   
# plt.xlabel(r"$\rm Resonance \ (THz)$", fontsize = '30')
# # plt.xlim(1000,2000)
# plt.ylim(minni,maxi)

# norm = mpl.colors.Normalize(vmin=0.42, vmax=0.85)
# pc_kwargs = {'rasterized': True, 'cmap': 'jet'}

# im = ax.pcolormesh(Y, X, Full, norm=norm, **pc_kwargs)

# plt.setp(ax.spines.values(), linewidth=2)
# plt.tick_params(left = False, bottom = False)
# cbar = fig.colorbar(im, ax=ax)
# cbar.set_label(label = r"$Reflectance$", size = '20')
# plt.savefig("HMFullDevRefRefMinFreq.pdf")
# plt.savefig("HMFullDevRefRefMinFreq.png")

###############################################################################

X = 1/(lam*1e-4)
Y = res
print(len(res))
print(len(peak))
print(len(Full))

minni = 1/(10.0*1e-4)
maxi  = 1/(4.5*1e-4)
dcc = 1/(dcc*1e-4)
peak = 1/(peak*1e-4)
peak2 = 1/(peak2*1e-4)
# Full = np.reshape(Full, (10,NFREQs), order='C')

fig, ax = plt.subplots(figsize=(9,12),constrained_layout=True)
plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(direction = 'in', width=2, labelsize=20)
plt.ylabel(r"$\rm Wavenumber \ (cm^{-1})$", fontsize = '30')   
plt.xlabel(r"$\rm Resonance  \ (cm^{-1})$", fontsize = '30')
plt.xlim(min(Y),max(Y))
plt.ylim(minni,maxi)

norm = mpl.colors.Normalize(vmin=0.42, vmax=0.85)
pc_kwargs = {'rasterized': True, 'cmap': 'jet'}

im = ax.pcolormesh(Y, X, Full, norm=norm, **pc_kwargs)

plt.scatter(Dots, peak,     linewidth = 3, s=55,edgecolors = 'white', c='red',  zorder = 25)  
# print(len(Y))     
# Y=np.delete(Y,11)
# peak2=np.delete(peak2,11)

# Y=np.delete(Y,11)
# peak2=np.delete(peak2,11)
# print(len(Y))     
# peak2 = 1/(peak2*1e-4)

plt.scatter(Dots, peak2,    linewidth = 3, s=55,edgecolors = 'white', c='blue', zorder = 25)       

plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label(label = r"$Reflectance$", size = '20')
plt.savefig("HMFullDevRefRefMinWN.pdf")
plt.savefig("HMFullDevRefRefMinWN.png")
###############################################################################
###########################################################################

# X = lam

# PitchLeng = np.linspace(2.02, 2.70, 18)
# dcc = np.zeros(numPlots, dtype = np.double)
# # dcc[0] = PitchLeng
# dcc[18] = 2.78
# Y = dcc

# # Full = np.transpose(Full)
# fig, ax = plt.subplots(figsize=(9,12),constrained_layout=True)
# plt.setp(ax.spines.values(), linewidth=2)
# plt.tick_params(direction = 'in', width=2, labelsize=20)
# plt.ylabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '30')   
# plt.xlabel(r"$\rm d_{cc}  \ (\mu m)$", fontsize = '30')
# # plt.xlim(1000,2000)
# plt.ylim(5,9)

# norm = mpl.colors.Normalize(vmin=0.42, vmax=0.85)
# pc_kwargs = {'rasterized': True, 'cmap': 'jet'}

# im = ax.pcolormesh(Y, X, Full, norm=norm, **pc_kwargs)

# plt.setp(ax.spines.values(), linewidth=2)
# plt.tick_params(left = False, bottom = False)
# cbar = fig.colorbar(im, ax=ax)
# cbar.set_label(label = r"$Reflectance$", size = '20')
# plt.savefig("HMWLAxis.pdf")
# plt.savefig("HMWLAxis.png")
