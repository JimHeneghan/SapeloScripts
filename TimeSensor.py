import numpy as np
from scipy import *
from pylab import *
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
d = 50e-9
imp0 = 376.730313
# k0 = k*1e12
c0 = 3e8
numpad = 10000000 



R = np.loadtxt("ExHeatMapTime.txt", usecols=(0), skiprows= 1, unpack =True )

fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(direction = 'in', width=2, labelsize=20)
plt.ylabel("R", fontsize = '30')   
plt.xlabel(r"$\rm \Delta t$", fontsize = '30')

plt.setp(ax.spines.values(), linewidth=2)
plt.plot(R[0: len(R)])
plt.savefig("RefhBN_AgSiSep23.png")


# T = np.loadtxt("ExRefTime.txt", usecols=(0), skiprows= 1, unpack =True )

# fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
# plt.setp(ax.spines.values(), linewidth=2)
# plt.tick_params(direction = 'in', width=2, labelsize=20)
# plt.ylabel("R", fontsize = '30')   
# plt.xlabel(r"$\rm \Delta t$", fontsize = '30')

# plt.setp(ax.spines.values(), linewidth=2)
# plt.plot(T[0: len(R)])
# plt.savefig("TrahBN_AgSiSep23.png")
# # plt.show()


fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(direction = 'in', width=2, labelsize=20)
plt.ylabel("R", fontsize = '30')   
plt.xlabel(r"$\rm \Delta t$", fontsize = '30')

plt.setp(ax.spines.values(), linewidth=2)
plt.plot(R[1500000: len(R)])
plt.savefig("Ref_skip2500k_to1_5M_hBN_AgSiSep23.png")