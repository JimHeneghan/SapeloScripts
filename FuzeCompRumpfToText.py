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
import matplotlib.patches as mpatches

cols = ["red", "gold", "green"]
mpl.rcParams['agg.path.chunksize'] = 10000
#all units are in m^-1
##############################################################################
fig, axs = plt.subplots(3,3, figsize=(22,22), sharex = True, sharey = 'row')
fig.subplots_adjust(hspace=0)
fig.subplots_adjust(wspace=0)
ax0 = fig.add_subplot(111, frameon=False)
ax0.set_xticks([])
ax0.set_yticks([])
ax0.set_xlabel(r'$\rm Frequency (cm^{-1})$', fontsize = '70', labelpad = 40)
##############################################################################
cover = ['FuzeBare','FuzeDead', 'FuzehBN']
dirk  = ['2.25um', '2.35um', '2.45um']
dirks = [2.25, 2.35, 2.45]
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
k = 0
for k in range(0,3):
	q = 0	
	for q in range(0,3):

		for i in range(0,3):
			# print(i)
			finder = "../%s/%s/Ref/GoldCross2_5umRTA.txt" %(cover[k], dirk[i])
			print("%s spectra for the %s simulation at dcc = %s " %(spec[q], cover[k], dirk[i]))
			R = np.loadtxt(finder,  usecols=(1), skiprows= 1, unpack =True)
			T = np.loadtxt(finder,  usecols=(2), skiprows= 1, unpack =True)
			A = np.loadtxt(finder,  usecols=(3), skiprows= 1, unpack =True)

			plt.setp(axs[q,k].spines.values(), linewidth=2)
			axs[q,k].tick_params(direction = 'in', width=2, length=5, labelsize=30)
			if (k==0):
				axs[q,k].set_ylabel(r"$%s$" %spec[q], fontsize = '45')   
			axs[q,k].set_xlim(600,3000)
			axs[q,k].set_ylim(0,1)
			if (q==0):
				axs[q,k].plot(1/(lam*1e-4), R, label = r'$\rm %s_{%s \ \mu m}$' %(spec[q], dirks[i]), color = cols[i], linewidth = 4)
			elif (q==1):
				axs[q,k].plot(1/(lam*1e-4), T, label = r'$\rm %s_{%s \ \mu m}$' %(spec[q], dirks[i]), color = cols[i], linewidth = 4)
			elif (q==2):
				axs[q,k].plot(1/(lam*1e-4), A, label = r'$\rm %s_{%s \ \mu m}$' %(spec[q], dirks[i]), color = cols[i], linewidth = 4)
				axs[q,k].set_ylim(0,0.45)

			# ax0.plot(1/(lam*1e-4), Tra, label = r'$\rm T_{%s}$' %dirk[q], color = "blue", linewidth = 3)
			# ax0.plot(1/(lam*1e-4), Abs, label = r'$\rm A_{%s}$' %dirk[q], color = "limegreen", linewidth = 3)

			# axs[q,k].tick_params( width=2, labelsize=30)   
			axs[q,k].yaxis.set_major_locator(MaxNLocator(nbins = 5,prune='lower'))
			axs[q,k].xaxis.set_major_locator(MaxNLocator(nbins = 4,prune='both'))
			# axs[q,k].legend(loc='center right', fontsize='35')

axs[0,0].set_title("Bare", fontsize = '45')
axs[0,1].set_title("Uncoupled", fontsize = '45')
axs[0,2].set_title("hBN", fontsize = '45')

patches = []
PitchLeng = np.linspace(2.02, 2.78, 20)
# dcc = np.zeros(20, dtype = np.double)
# dcc = PitchLeng
dcc = [2.10, 2.38, 2.58]
for i in range(0, 3):
    temp = mpatches.Patch(facecolor=cols[i], label = r'$ d_{cc} \rm = %2.2f \ \mu m$' %dirks[i], edgecolor='black')
    patches.append(temp) 
leg = ax0.legend(handles = patches, ncol = 3, loc = 'lower center', frameon = True,fancybox = False, 
fontsize = 30, bbox_to_anchor=(0, 1.05, 1, .175),mode="expand", borderaxespad=0.) #bbox_to_anchor=(1.05, 0.5),
leg.get_frame().set_edgecolor('black')

leg.get_frame().set_linewidth(2)
# axs[2,0].ylim(0,0.2)
# axs[2,1].ylim(0,0.2)			
# axs[2,2].set_ylim(0,0.2)
# axs.set_xlabel(r"$\rm Frequency \ (cm^{-1})$", fontsize = '45')

plt.savefig("FuzeAllRTA.png")

