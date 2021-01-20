import numpy as np
from scipy import *
from pylab import *
import math
import cmath
from ctypes import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib import colors as c

cols = ["red", "gold", "green"]
mpl.rcParams['agg.path.chunksize'] = 10000
#all units are in m^-1
##############################################################################
fig, axs = plt.subplots(3, figsize=(12,22),constrained_layout=True, sharex = True)

##############################################################################
dirk  = ['2.25um', '2.35um', '2.45um']
dirks = ['2.25', '2.35', '2.45']
spec =  ["R", "T", "A"]
##############################################################################

dx     = 20e-9
dy     = 20e-9
c0     = 3e8
nref   = 1.0
ntra   = 3.42
NFREQs = 500


###########################################################################
lam        = np.loadtxt("2.45um/Ref/GoldCross2_5umRTA.txt",  usecols=(0), skiprows= 1, unpack =True)

###########################################################################

###########################################################################
i = 0
j = 0
q = 0
for ax0 in axs.flat:

	for i in range(0,3):
		print(i)
		print(j)
		finder = "%s/Ref/GoldCross2_5umRTA.txt" %dirk[i]
		R = np.loadtxt(finder,  usecols=(1), skiprows= 1, unpack =True)
		T = np.loadtxt(finder,  usecols=(2), skiprows= 1, unpack =True)
		A = np.loadtxt(finder,  usecols=(3), skiprows= 1, unpack =True)

		plt.setp(ax0.spines.values(), linewidth=2)
		ax0.tick_params(direction = 'in', width=2, labelsize=20)
		ax0.set_ylabel(r"$%s$" %spec[q], fontsize = '45')   
		ax0.set_xlim(600,3000)

		if (q==0):
			ax0.plot(1/(lam*1e-4), R, label = r'$\rm %s_{%s \ \mu m}$' %(spec[q], dirks[i]), color = cols[i], linewidth = 4)
			ax0.set_ylim(0,1)
		if (q==1):
			ax0.plot(1/(lam*1e-4), T, label = r'$\rm %s_{%s \ \mu m}$' %(spec[q], dirks[i]), color = cols[i], linewidth = 4)
			ax0.set_ylim(0,1)
		if (q==2):
			ax0.plot(1/(lam*1e-4), A, label = r'$\rm %s_{%s \ \mu m}$' %(spec[q], dirks[i]), color = cols[i], linewidth = 4)
			ax0.set_ylim(0,0.2)
		# ax0.plot(1/(lam*1e-4), Tra, label = r'$\rm T_{%s}$' %dirk[q], color = "blue", linewidth = 3)
		# ax0.plot(1/(lam*1e-4), Abs, label = r'$\rm A_{%s}$' %dirk[q], color = "limegreen", linewidth = 3)

		ax0.tick_params(left = False, bottom = False, width=2, labelsize=30)   
		ax0.yaxis.set_major_locator(MaxNLocator(nbins = 5,prune='lower'))
		ax0.legend(loc='center right', fontsize='35')
	q = q+1
ax0.set_xlabel(r"$\rm Frequency \ (cm^{-1})$", fontsize = '45')

plt.savefig("GoldCrossSweepFuzeDeadRTA.png")

