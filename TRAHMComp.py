import numpy as np
from scipy import *
from pylab import *
import math
import cmath
from ctypes import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import UnivariateSpline

mpl.rcParams['agg.path.chunksize'] = 10000
#all units are in m^-1

# res = [6.598003, 6.724, 6.885993, 7.029992, 7.173996, 7.299998, 7.426, 7.570004, 7.677995, 7.821992]
# res = [5.94,  6.21,6.35, 6.48, 6.62, 6.75, 6.89, 7.03, 7.16, 7.30, 7.43, 7.57, 7.71, 7.84,  8.11]

AgRef    = [1650.7100777 , 1612.3833622 , 1584.78655619, 1549.42671212,
         1515.61010203, 1487.20999405, 1459.85337525, 1433.48664951,
         1404.49339572, 1376.653309  , 1356.48325747, 1333.69005097,
         1308.55659926, 1293.32660286, 1269.68004063, 1249.68664107,
         1227.59666771, 1214.18001991, 1193.31671043, 1175.64003313,  1155.]
resR = np.asarray(AgRef)
AgTra    = [1670.56325546, 1636.12672521, 1607.71678333, 1575.79677012,
       1545.11671271, 1519.75660794, 1495.21665241, 1467.56677429,
       1440.92343596, 1415.22664996, 1393.92327512, 1369.863389  ,
       1346.61998384, 1330.49335093, 1308.55659926, 1287.33330965,
       1266.78329518, 1249.68664107, 1230.31329559, 1214.18001991, 1214.0]
res = np.asarray(AgTra)
AgAbs    = [1670.56325546, 1631.32003971, 1607.71678333, 1575.79677012,
       1545.11671271, 1511.48661813, 1483.24005484, 1448.43673844,
       1418.84343261, 1390.43342869, 1366.49339075, 1343.36342199,
       1317.8699742 , 1299.37663705, 1275.51004139, 1255.33659277,
       1233.04668697, 1216.84004406, 1198.46668173, 1180.63670793,  1155.64003313]
resA = np.asarray(AgAbs)

Dots = [1650.7100777 , 1612.3833622 , 1584.78655619, 1549.42671212,
       1515.61010203, 1487.20999405, 1459.85337525, 1433.48664951,
       1404.49339572, 1376.653309  , 1356.48325747, 1333.69005097,
       1308.55659926, 1293.32660286, 1269.68004063, 1249.68664107,
       1227.59666771, 1214.18001991, 1193.31671043, 1155.64003313,]

DeadRRes = [1640.95670401, 1603.07662466, 1571.34001519, 1536.56990271,
       1507.38664679, 1475.35993988, 1452.22337577, 1422.47672544,
       1393.92327512, 1369.863389  , 1349.89328419, 1327.31669524,
       1302.42335401, 1284.35670233, 1263.90341149, 1244.08995069,
       1222.19670547, 1208.89660863, 1190.75998827, 1170.68670726]

DeadTRes = [1670.56325546, 1636.12672521, 1607.71678333, 1575.79677012,
       1545.11671271, 1519.75660794, 1495.21665241, 1467.56677429,
       1440.92343596, 1415.22664996, 1393.92327512, 1369.863389  ,
       1346.61998384, 1330.49335093, 1308.55659926, 1287.33330965,
       1266.78329518, 1249.68664107, 1230.31329559, 1214.18001991]

DeadARes = [1670.56325546, 1631.32003971, 1607.71678333, 1575.79677012,
       1545.11671271, 1499.24992526, 1467.56677429, 1437.19335683,
       1408.05327623, 1380.07328741, 1356.48325747, 1333.69005097,
       1308.55659926, 1290.32341311, 1269.68004063, 1246.88325943,
       1227.59666771, 1211.53336145, 1193.31671043, 1175.64003313]

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
TRA2_02um  = np.loadtxt("2_02um/Ref/2_02umPitchRTA.txt",  usecols=(2), skiprows= 1, unpack =True)
A2_02um    = np.loadtxt("2_02um/Ref/2_02umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
Full.append(TRA2_02um)
###########################  Pitch = **** um ##############################
TRA2_06um  = np.loadtxt("2_06um/Ref/2_06umPitchRTA.txt",  usecols=(2), skiprows= 1, unpack =True)
A2_06um    = np.loadtxt("2_06um/Ref/2_06umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
Full.append(TRA2_06um)
###########################  Pitch = **** um ##############################
TRA2_10um  = np.loadtxt("2_10um/Ref/2_10umPitchRTA.txt",  usecols=(2), skiprows= 1, unpack =True)
A2_10um    = np.loadtxt("2_10um/Ref/2_10umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
Full.append(TRA2_10um)

###########################  Pitch = **** um ##############################
TRA2_14um  = np.loadtxt("2_14um/Ref/2_14umPitchRTA.txt",  usecols=(2), skiprows= 1, unpack =True)
A2_14um    = np.loadtxt("2_14um/Ref/2_14umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
Full.append(TRA2_14um)

###########################  Pitch = **** um ##############################
TRA2_18um  = np.loadtxt("2_18um/Ref/2_18umPitchRTA.txt",  usecols=(2), skiprows= 1, unpack =True)
A2_18um    = np.loadtxt("2_18um/Ref/2_18umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
Full.append(TRA2_18um)

###########################  Pitch = **** um ##############################
TRA2_22um  = np.loadtxt("2_22um/Ref/2_22umPitchRTA.txt",  usecols=(2), skiprows= 1, unpack =True)
A2_22um    = np.loadtxt("2_22um/Ref/2_22umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
Full.append(TRA2_22um)
###########################  Pitch = **** um ##############################
TRA2_26um  = np.loadtxt("2_26um/Ref/2_26umPitchRTA.txt",  usecols=(2), skiprows= 1, unpack =True)
A2_26um    = np.loadtxt("2_26um/Ref/2_26umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
# Full[,1]    = TRA2_26um
Full.append(TRA2_26um)

###########################################################################
TRA2_30um  = np.loadtxt("2_30um/Ref/2_30umPitchRTA.txt",  usecols=(2), skiprows= 1, unpack =True)
A2_30um    = np.loadtxt("2_30um/Ref/2_30umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
# Full[,2]    = TRA2_30um
Full.append(TRA2_30um)

###########################################################################
TRA2_34um  = np.loadtxt("2_34um/Ref/2_34umPitchRTA.txt",  usecols=(2), skiprows= 1, unpack =True)
A2_34um    = np.loadtxt("2_34um/Ref/2_34umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
Full.append(TRA2_34um)

# Full[,3]    = TRA2_34um
###########################################################################
TRA2_38um = np.loadtxt("2_38um/Ref/2_38umPitchRTA.txt",  usecols=(2), skiprows= 1, unpack =True)
A2_38um   = np.loadtxt("2_38um/Ref/2_38umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
Full.append(TRA2_38um)

# Full[4]    = TRA2_38um
###########################################################################
TRA2_42um = np.loadtxt("2_42um/Ref/2_42umPitchRTA.txt",  usecols=(2), skiprows= 1, unpack =True)
A2_42um   = np.loadtxt("2_42um/Ref/2_42umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
Full.append(TRA2_42um)

# Full[5]    = TRA2_42um
###########################################################################
TRA2_46um = np.loadtxt("2_46um/Ref/2_46umPitchRTA.txt",  usecols=(2), skiprows= 1, unpack =True)
A2_46um   = np.loadtxt("2_46um/Ref/2_46umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
Full.append(TRA2_46um)
# Full[6]    = TRA2_46um
###########################################################################
TRA2_50um = np.loadtxt("2_50um/Ref/2_50umPitchRTA.txt",  usecols=(2), skiprows= 1, unpack =True)
A2_50um   = np.loadtxt("2_50um/Ref/2_50umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
Full.append(TRA2_50um)
# Full[7]    = TRA2_50um
###########################################################################
TRA2_54um = np.loadtxt("2_54um/Ref/2_54umPitchRTA.txt",  usecols=(2), skiprows= 1, unpack =True)
A2_54um   = np.loadtxt("2_54um/Ref/2_54umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
Full.append(TRA2_54um)
# Full[8]    = TRA2_54um
###########################################################################
TRA2_58um = np.loadtxt("2_58um/Ref/2_58umPitchRTA.txt",    usecols=(2), skiprows= 1, unpack =True)
A2_58um   = np.loadtxt("2_58um/Ref/2_58umPitchRTA.txt",    usecols=(3), skiprows= 1, unpack =True)
Full.append(TRA2_58um)
###########################################################################
TRA2_62um = np.loadtxt("2_62um/Ref/2_62umPitchRTA.txt",    usecols=(2), skiprows= 1, unpack =True)
A2_62um   = np.loadtxt("2_62um/Ref/2_62umPitchRTA.txt",    usecols=(3), skiprows= 1, unpack =True)
Full.append(TRA2_62um)
###########################################################################
TRA2_66um = np.loadtxt("2_66um/Ref/2_66umPitchRTA.txt",    usecols=(2), skiprows= 1, unpack =True)
A2_66um   = np.loadtxt("2_66um/Ref/2_66umPitchRTA.txt",    usecols=(3), skiprows= 1, unpack =True)
Full.append(TRA2_66um)
###########################################################################
TRA2_70um = np.loadtxt("2_70um/Ref/2_70umPitchRTA.txt",    usecols=(2), skiprows= 1, unpack =True)
A2_70um   = np.loadtxt("2_70um/Ref/2_70umPitchRTA.txt",    usecols=(3), skiprows= 1, unpack =True)
Full.append(TRA2_70um)
###########################################################################
TRA2_74um = np.loadtxt("2_74um/Ref/2_74umPitchRTA.txt",    usecols=(2), skiprows= 1, unpack =True)
A2_74um   = np.loadtxt("2_74um/Ref/2_74umPitchRTA.txt",    usecols=(3), skiprows= 1, unpack =True)
Full.append(TRA2_74um)
###########################################################################
TRA2_78um = np.loadtxt("2_78um/Ref/2_78umPitchRTA.txt",    usecols=(2), skiprows= 1, unpack =True)
A2_78um   = np.loadtxt("2_78um/Ref/2_78umPitchRTA.txt",    usecols=(3), skiprows= 1, unpack =True)
Full.append(TRA2_78um)
###########################################################################
numPlots = 20
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
for i in range(0, numPlots):
    for j in range(0,NFREQs-1):
        if ((pitchThou[i,j] < pitchThou[i,j-1]) & (pitchThou[i,j+1] > pitchThou[i,j]) & ((1/(lam[j]*1e-4) < 2000)) & ((1/(lam[j]*1e-4) > 1000))):
            TraTroughLam[i]  = (1/(lam[j]*1e-4))
            # print(j)
            TraTrough[i]     = pitchThou[i,j]


for i in range(0, numPlots):
    temp = 1.0
    breaker = 0
    for j in range(0,NFREQs-1):
        if ((pitchThou[i,j] < pitchThou[i,j-1]) & (pitchThou[i,j+1] > pitchThou[i,j]) & (temp > pitchThou[i,j]) & ((1/(lam[j]*1e-4) > TraTroughLam[i]))):
            temp = pitchThou[i,j]
            TraTroughLam2[i] = (1/(lam[j]*1e-4))
            # print(j)
            TraTrough2[i]    = pitchThou[i,j]
            breaker = 1

# for i in range(5, numPlots):
#     temp = 1.0
#     breaker = 0
#     for j in range(250,NFREQs-1):
#         if ((pitchThou[i,j] < pitchThou[i,j-1]) & (pitchThou[i,j+1] > pitchThou[i,j])  & ((1/(lam[j]*1e-4) > TraTroughLam[i]))&(breaker==0)):
#             temp = pitchThou[i,j]
#             TraTroughLam2[i] = (1/(lam[j]*1e-4))
#             # print(j)
#             TraTrough2[i]    = pitchThou[i,j]
#             breaker = 1

# for i in range(0, 2):
#     temp = 1.0
#     breaker = 0
#     for j in range(250,NFREQs-1):
#         if ((pitchThou[i,j] < pitchThou[i,j-1]) & (pitchThou[i,j+1] > pitchThou[i,j])  & ((1/(lam[j]*1e-4) > TraTroughLam[i]))):
#             temp = pitchThou[i,j]
#             TraTroughLam2[i] = (1/(lam[j]*1e-4))
#             # print(j)
#             TraTrough2[i]    = pitchThou[i,j]
#             breaker = 1
###############################################################################

# peak = np.asarray(peak)

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
# plt.xlabel(r"$\rm \nu_{Bare} \ (THz)$", fontsize = '30')
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
print(len(X))
print(len(Full))

minni = 1/(10.0*1e-4)
maxi  = 1/(4.5*1e-4)
dcc = 1/(dcc*1e-4)
# peak = 1/(peak*1e-4)
# peak2 = 1/(peak2*1e-4)
# Full = np.reshape(Full, (10,NFREQs), order='C')

fig, ax = plt.subplots(figsize=(9,12),constrained_layout=True)
plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(direction = 'in', width=2, labelsize=20)
plt.ylabel(r"$\rm Frequency \ (cm^{-1})$", fontsize = '30')   
plt.xlabel(r"$\rm \nu_{Bare}  \ (cm^{-1})$", fontsize = '30')
plt.xlim(min(Y),max(Y))
plt.ylim(1100,1700)

norm = mpl.colors.Normalize(vmin=0, vmax=0.38)
pc_kwargs = {'rasterized': True, 'cmap': 'jet_r'}

im = ax.pcolormesh(Y, X, Full, norm=norm, **pc_kwargs)

# plt.scatter(AgTra[0:numPlots], DeadTRes, linewidth = 2, s=150,edgecolors = 'white', c='limegreen', label = "Dead Resonance")       

# plt.scatter(AgTra[0:numPlots], TraTroughLam2,  linewidth = 2, s=55,edgecolors = 'white', c='blue', label = "Polariton")       
plt.scatter(AgTra[0:numPlots], TraTroughLam, linewidth = 5, s=70,edgecolors = 'black', c='gold', label = "Plasmon")       

plt.scatter(AgTra[0:numPlots], TraTroughLam, linewidth = 2, s=45,edgecolors = 'white', c='gold', label = "Plasmon")       

plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label(label = r"$\rm Transmittance $", size = '30')
cbar.ax.tick_params(labelsize=20) 

plt.savefig("TraDeadRasterHM.pdf")
plt.savefig("TraDeadRasterHM.png")
###############################################################################
peak  = np.zeros(numPlots, dtype = np.double)
peakA = np.zeros(numPlots, dtype = np.double)

peak2  = np.zeros(numPlots, dtype = np.double)
peakA2 = np.zeros(numPlots, dtype = np.double)
###########################################################################
pitchThou = np.zeros((numPlots,NFREQs), dtype = np.double)

Full = []
pitchThou[0]   = A2_02um
pitchThou[1]   = A2_06um
pitchThou[2]   = A2_10um
pitchThou[3]   = A2_14um
pitchThou[4]   = A2_18um
pitchThou[5]   = A2_22um
pitchThou[6]   = A2_26um
pitchThou[7]   = A2_30um
pitchThou[8]   = A2_34um
pitchThou[9]   = A2_38um
pitchThou[10]  = A2_42um
pitchThou[11]  = A2_46um
pitchThou[12]  = A2_50um
pitchThou[13]  = A2_54um
pitchThou[14]  = A2_58um
pitchThou[15]  = A2_62um
pitchThou[16]  = A2_66um
pitchThou[17]  = A2_70um
pitchThou[18]  = A2_74um
pitchThou[19]  = A2_78um

Full.append(A2_02um)
###########################  Pitch = **** um ##############################
Full.append(A2_06um)
Full.append(A2_10um)
Full.append(A2_14um)

Full.append(A2_18um)

Full.append(A2_22um)

Full.append(A2_26um)

Full.append(A2_30um)


Full.append(A2_34um)

Full.append(A2_38um)


Full.append(A2_42um)

Full.append(A2_46um)
# Full[6]    = A2_46um

Full.append(A2_50um)
# Full[7]    = A2_50um

Full.append(A2_54um)
# Full[8]    = A2_54um

Full.append(A2_58um)

Full.append(A2_62um)

Full.append(A2_66um)

Full.append(A2_70um)

Full.append(A2_74um)
Full.append(A2_78um)
###############################################################################
for i in range(0, numPlots):
    for j in range(0,NFREQs-1):
        if ((pitchThou[i,j] > pitchThou[i,j-1]) & (pitchThou[i,j+1] < pitchThou[i,j]) & ((1/(lam[j]*1e-4) < 2000)) & ((1/(lam[j]*1e-4) > 1000))):
            peak[i]  = (1/(lam[j]*1e-4))
            # print(j)
            peakA[i] = pitchThou[i,j]


for i in range(0, numPlots):
    temp = 0.0
    for j in range(250,NFREQs-1):
        if ((pitchThou[i,j] > pitchThou[i,j-1]) & (pitchThou[i,j+1] < pitchThou[i,j]) & (temp < pitchThou[i,j]) & ((1/(lam[j]*1e-4) > peak[i]))):
            temp = pitchThou[i,j]
            peak2[i]  = (1/(lam[j]*1e-4))
            # print(j)
            peakA2[i] = pitchThou[i,j]
# for i in range(10, 13):
#     y_spl = UnivariateSpline(lam[334:355],pitchThou[i,334:355],s=0,k=5)
#     # plt.plot(lam[334:360],pitchThou[i,334:360], 'ro')
#     x_range = np.linspace(lam[334],lam[355],1000)
#     # plt.semilogy(x_range,y_spl(x_range))
#     y_spl_2d = y_spl.derivative(n=2)

#     plt.plot(x_range,y_spl_2d(x_range))
#     # print(y_spl_2d.roots())
#     # print(y_spl_2d(x_range))
#     # rootys = y_spl_2d.roots()
#     # X0s = np.zeros(len(rootys), dtype = np.double)
#     # plt.semilogy(rootys,y_spl(rootys), 'bo')
#     y_spl = y_spl(x_range)

#     y_spl_2d = y_spl_2d(x_range)
#     temp1 = 0.0
#     for j in range(0,998):
#       if((y_spl_2d[j-1] > y_spl_2d[j]) & (y_spl_2d[j] < y_spl_2d[j+1]) & (y_spl_2d[j] < temp1)):
#         peak2[i]  = (1/(x_range[j]*1e-4))
#         peakA2[i] = y_spl[j] 
#         temp1 = y_spl_2d[j]
#         temp2 = y_spl_2d[j]
###############################################################################

PitchLeng = np.linspace(2.02, 2.78, 20)
dcc = np.zeros(numPlots, dtype = np.double)
dcc = PitchLeng
dcc1 = dcc
###############################################################################
Full = np.transpose(Full)


###############################################################################

X = 1/(lam*1e-4)
Y = resA
print(len(res))
print(len(X))
print(len(Full))

minni = 1/(10.0*1e-4)
maxi  = 1/(4.5*1e-4)
dcc = 1/(dcc*1e-4)
# peak = 1/(peak*1e-4)
# peak2 = 1/(peak2*1e-4)
# Full = np.reshape(Full, (10,NFREQs), order='C')

fig, ax = plt.subplots(figsize=(9,12),constrained_layout=True)
plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(direction = 'in', width=2, labelsize=20)
plt.ylabel(r"$\rm Frequency \ (cm^{-1})$", fontsize = '30')   
plt.xlabel(r"$\rm \nu_{Bare}  \ (cm^{-1})$", fontsize = '30')
plt.xlim(min(Y),max(Y))
plt.ylim(1100,1700)

norm = mpl.colors.Normalize(vmin=0.0, vmax=0.30)
pc_kwargs = {'rasterized': True, 'cmap': 'jet'}

im = ax.pcolormesh(Y, X, Full, norm=norm, **pc_kwargs)

# plt.scatter(AgAbs[0:numPlots], DeadARes, linewidth = 2, s=150,edgecolors = 'white', c='limegreen', label = "Dead Resonance")       

# plt.scatter(AgAbs[0:numPlots], peak2,  linewidth = 2, s=55,edgecolors = 'white', c='blue', label = "Polariton")       
plt.scatter(AgAbs[0:numPlots], peak, linewidth = 5, s=70,edgecolors = 'black', c='gold', label = "Plasmon")       

plt.scatter(AgAbs[0:numPlots], peak, linewidth = 2, s=45,edgecolors = 'white', c='gold', label = "Plasmon")       

plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label(label = r"$\rm Absorptance$", size = '30')
cbar.ax.tick_params(labelsize=20) 

plt.savefig("AbsDeadRasterHM.pdf")
plt.savefig("AbsDeadRasterHM.png")



