#!/usr/local/apps/anaconda/3-2.2.0/bin/python
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
Nz = 500
Ny = 41
fig, ax0 = plt.subplots(figsize = (5, 3),constrained_layout=True)
Frame = []
for z in range(0, 60):
	#print(z)
	Field = "MovieXZ/Ex%d.dat" %(3*z)
	Full = np.zeros((Nz, Ny), dtype = np.double)

	X = np.linspace(0,  Ny, Ny)
	Y = np.linspace(0,  Nz, Nz)
	
	E = np.loadtxt(Field, usecols=(0,), skiprows= 0, unpack =True )
	for i in range (0, Nz):
	 	Full[i] = E[i*Ny: i*Ny + Ny]

	print(max(E))
	print(min(E))
	norm = mpl.colors.Normalize(vmin=-1, vmax=1)
	pc_kwargs = {'rasterized': True, 'cmap': 'jet'}
	
	#ax0.set_aspect(100/500)
	
	im = ax0.pcolormesh(X, Y, Full, norm=norm, **pc_kwargs)
	# ax0.axhline(y =50, color = 'silver', linewidth=1)

	# ax0.axhline(y =150, color = 'fuchsia', linewidth=1)
	# ax0.axhline(y =350, color = 'fuchsia', linewidth=1)

	# ax0.axhline(y =225, color = 'silver', linewidth=1)
	# ax0.axhline(y =250, color = 'silver', linewidth=1)

	# ax0.axhline(y =160, color = 'yellow', linewidth=1)

	cbar = fig.colorbar(im, ax=ax0)
	cbar.set_label(label = r"$\rm E \ \ Field \ (V m^{-1})$", size = '20')

	#ax0.set_title(r"$ \rm  %.2f \ (\mu m)$" % fontsize = '25')
	ax0.set_xlabel(r"$ \rm x \ $", fontsize = '20')
	ax0.set_ylabel(r"$ \rm y \ $", fontsize = '20')

	plt.setp(ax0.spines.values(), linewidth=2)
	plt.tick_params(left = False, bottom = False)

	Frame.append([im])
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)


#fig1 = plt.subplots(figsize = (10, 2),constrained_layout=True)
anim = animation.ArtistAnimation(fig, Frame, interval = 20)
#plt.show()
anim.save("YZFull.mp4", writer = writer )
	#plt.savefig("hBN_Ag_Si%dHz.pdf" %z)
	#plt.savefig("../EdgeImages/DomainWall/EdgeEz%d.png" %z)

