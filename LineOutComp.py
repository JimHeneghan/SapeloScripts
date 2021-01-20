#!/usr/local/apps/anaconda/3-2.2.0/bin/python
import time 
import numpy as np
import scipy as sp
import itertools as it
import math
import collections as cl
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib_scalebar.scalebar import ScaleBar, SI_LENGTH
import matplotlib.patches as mpatches

#______________________________________________________________________#
############################## |E| Field ###############################
#______________________________________________________________________#

fig, (ax2, ax3)= plt.subplots(1,2, figsize = (18,12))#, constrained_layout=True)
# plt.gcf().subplots_adjust(left = 0.05, right = 0.99, top = 0.97, bottom = -0.1)

# mpl.rcParams['axes.linewidth'] = 10
Frame = []
Nx = 129
Ny = 213
z = 0
dt = 5.945e-18
T = 2e6
#print(z)
shades = ['red', 'limegreen', 'blue', 'purple', 'magenta', 'red', 'orange', 'gold', 'green', 'blue', 'purple', 'magenta']

freq = ['6.1', '6.3', '6.5', '6.7', '6.8', '6.9', '7.0', '7.1', '7.2', '7.5', '8.0', '9.0', '9.5', '10.0', '10.5']
selectionBoxAli = [1, 2, 5, 6, 8, 9]
selectionBoxJim = [1, 2, 3, 4, 5, 6, 8, 9]

ShadesRed  = ['deeppink','fuchsia', 'purple', 'darkviolet', 'blue', 'dodgerblue', 'deepskyblue', 'teal', 'springgreen', 'seagreen', 'limegreen',
 'forestgreen', 'greenyellow','gold', 'orange', 'orangered', 'salmon', 'red', 'darkred', 'lightcoral']

sboxAli = [20, 11, 9, 5]


for z in range(0, 4):

	base  = 6.0 + 0.1*sboxAli[z]
	Field = "../../AliData/%s_um.txt" %base
	# z = z
	x, E = np.loadtxt(Field, usecols=(0,1), skiprows= 3, unpack =True )
	x = x - max(x)/2
	print(min(E))
	print(max(E))
	# if (z == 2):
	# 	E = E + 0.15
	# mult = 0.4
	# if (z > 1):
	# 	mult = 0.25

	if ((z == 1) or (z == 2)):
		ax2.plot(x*1e6, E  - E[0] + 0.25*z, linewidth=2, color = shades[z], label = r"$ \rm %s \ \mu m$" %freq[z])
	if ((z == 3)):
		ax2.plot(x*1e6, 2*E  - 2*E[0] + 0.25*z, linewidth=2, color = shades[z], label = r"$ \rm %s \ \mu m$" %freq[z])
	if ((z == 0)):
		ax2.plot(x*1e6, 4*E  - 4*E[0] + 0.25*z, linewidth=2, color = shades[z], label = r"$ \rm %s \ \mu m$" %freq[z])


	#plt.savefig("EFieldXSec43THz.pdf")
ax2.axvline(x = -0.68, linestyle = "dashed", color = 'black')
ax2.axvline(x =  0.68, linestyle = "dashed", color = 'black')
ax2.set_xlim(-1.29, 1.29)
ax2.set_ylim(-0.1, 1.05)
# plt.setp(ax2.spines.values(), linewidth=1)
ax2.tick_params(left = False, bottom = False,labelsize = '15', direction = 'in', pad = 15)
ax2.set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '30')
ax2.set_ylabel(r"$\rm s-SNOM \ Signal \ (arb. units)$", fontsize = '30')
for axis in ['top','bottom','left','right']:
  ax2.spines[axis].set_linewidth(4)
Nx = 129
Ny = 213
###############################################################
sboxJim = [190, 110, 90, 50]

for z in range(0, 4):
	lam = 6.0 + 0.01*sboxJim[z] # 6.0 + 0.1*sbox 
	# Field = "%s_um.txt" %freq[z]
	Fieldx = "Raw/ExHeatMap%.2f.txt" %lam
	Fieldy = "Raw/EyHeatMap%.2f.txt" %lam
	Fieldz = "Raw/EzHeatMap%.2f.txt" %lam 

	Full = np.zeros((Ny, Nx), dtype = np.double)
	ExAb = np.zeros((Ny, Nx), dtype = np.double)
	EyAb = np.zeros((Ny, Nx), dtype = np.double)
	EzAb = np.zeros((Ny, Nx), dtype = np.double)

	Y = np.linspace(0,  Ny, Ny)
	X = np.linspace(-1.29,  1.29, Nx)

	Ex = np.loadtxt(Fieldx, usecols=(0,), skiprows= 1, unpack =True )
	Ey = np.loadtxt(Fieldy, usecols=(0,), skiprows= 1, unpack =True )
	Ez = np.loadtxt(Fieldz, usecols=(0,), skiprows= 1, unpack =True )


	Ex = abs(Ex)*abs(Ex)*(dt/T)*(dt/T)
	Ey = abs(Ey)*abs(Ey)*(dt/T)*(dt/T)
	Ez = abs(Ez)*abs(Ez)*(dt/T)*(dt/T)

	for i in range (0, Ny):
		Full[i] = Ex[i*Nx: i*Nx + Nx] + Ey[i*Nx: i*Nx + Nx]+ Ez[i*Nx: i*Nx + Nx]
		Full[i] = np.sqrt(Full[i])

	print(min(1e20*Full[107]))
	print(max(1e20*Full[107]))
	ax3.plot(X, 1e20*(Full[107] - Full[107,0]) + 0.25*z, linewidth=2, color = shades[z], label = r"$ \rm %s \ \mu m$" %freq[z])



ax3.axvline(x = -0.68, linestyle = "dashed", color = 'black')
ax3.axvline(x =  0.68, linestyle = "dashed", color = 'black')
ax3.set_xlim(-1.29, 1.29)
ax3.set_ylim(-0.1, 1.05)

# plt.setp(ax3.spines.values(), linewidth=1)
ax3.tick_params(left = False, bottom = False,labelsize = '15', direction = 'in', pad = 15)
for axis in ['top','bottom','left','right']:
  ax3.spines[axis].set_linewidth(4)

ax3.set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '30')
ax3.set_ylabel(r"$\rm \vert E \vert \ Field \ (arb. units)$", fontsize = '30')
#################################################################################3
patches = []

ax0 = fig.add_subplot(111, frameon=False)
ax0.set_xticks([])
ax0.set_yticks([])
for i in range(0, 4):
	base  = 6.0 + 0.1*sboxAli[i]
	temp = mpatches.Patch(facecolor=shades[i], label = r"$\rm \nu= %2.0f \ cm^{-1} $" %(1/(base*1e-4)), edgecolor='black')
	patches.append(temp) 
leg = ax0.legend(handles = patches, ncol = 4, loc = 'lower center', frameon = True,fancybox = False, 
fontsize = 15, bbox_to_anchor=(0.0, -0.25, 1.0, .05),mode="expand", borderaxespad=0.) #bbox_to_anchor=(1.05, 0.5),
leg.get_frame().set_edgecolor('black')

leg.get_frame().set_linewidth(4)
plt.tight_layout()
plt.savefig("2PanelLineOut3.png")
plt.savefig("2PanelLineOut3.pdf")