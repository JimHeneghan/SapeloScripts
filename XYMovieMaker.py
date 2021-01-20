#!/usr/bin/env python
import time 
import numpy as np
import scipy as sp
import itertools as it
import math
import collections as cl
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
#from mpl_toolkits.axes_grid1 import make_axes_locatable

#many = np.zeros((10, 401, 231), dtype = np.double)
Nx = 41
Ny = 41
fig, ax0 = plt.subplots(figsize = (3, 3),constrained_layout=True)
Frame = []
for z in range(50, 75):
	#print(z)
	Field = "XYMovie/Ex%d.dat" %(z)
	Full = np.zeros((41, 41), dtype = np.double)

	X = np.linspace(0,  41, 41)
	Y = np.linspace(0,  41, 41)
	
	E = np.loadtxt(Field, usecols=(0,), skiprows= 0, unpack =True )
	for i in range (0, Ny):
	 	Full[i] = E[i*Nx: i*Nx + Nx]

	print(max(E))
	print(z)
	norm = mpl.colors.Normalize(vmin=0, vmax=0.1)
	pc_kwargs = {'rasterized': True, 'cmap': 'jet'}
	
	ax0.set_aspect(4/2.3)
	
	im = ax0.pcolormesh(X, Y, Full, norm=norm, **pc_kwargs)
	# ax0.axhline(y =115, color = 'fuchsia', linewidth=1)
	# ax0.axhline(y =415, color = 'fuchsia', linewidth=1)

	# ax0.axhline(y =265, color = 'silver', linewidth=1)
	# ax0.axhline(y =290, color = 'silver', linewidth=1)

	# ax0.axhline(y =125, color = 'yellow', linewidth=1)

	cbar = fig.colorbar(im, ax=ax0)
	cbar.set_label(label = r"$\rm E \ \ Field \ (V m^{-1})$", size = '20')

	#ax0.set_title(r"$ \rm  %.2f \ (\mu m)$" % fontsize = '25')
	ax0.set_xlabel(r"$ \rm x \ $", fontsize = '20')
	ax0.set_ylabel(r"$ \rm y \ $", fontsize = '20')

	plt.setp(ax0.spines.values(), linewidth=2)
	plt.tick_params(left = False, bottom = False)

	Frame.append([im])
	# plt.savefig("Images/Ex%s.png" %z)


Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)


#fig1 = plt.subplots(figsize = (10, 2),constrained_layout=True)
anim = animation.ArtistAnimation(fig, Frame, interval = 20)
#plt.show()
anim.save("XYToyTest.mp4", writer = writer )
	#plt.savefig("hBN_Ag_Si%dHz.pdf" %z)
	#plt.savefig("../EdgeImages/DomainWall/EdgeEz%d.png" %z)

