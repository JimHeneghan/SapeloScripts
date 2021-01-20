#!/usr/local/apps/anaconda/3-2.2.0/bin/python
import time 
import numpy as np
import scipy as sp
import itertools as it
import math
import collections as cl
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspeceps0Off
from matplotlib.animation import FuncAnimation
#from mpl_toolkits.axes_grid1 import make_axes_locatable

#many = np.zeros((10, 401, 231), dtype = np.double)
Nz = 1000
Nx = 45
T = 600
fig, ax0 = plt.subplots(figsize = (5, 5),constrained_layout=True)
Full = np.zeros((T, Nz, Nx), dtype = np.double)
for z in range(0, T):
	
	fr = z
	print(fr)
	Field = "ZXMovie/Ex%d.dat" %(fr)
	Frame = np.zeros((Nz, Nx), dtype = np.double)

	X = np.linspace(0,  Nx, Nx)
	Y = np.linspace(0,  Nz, Nz)

	E = np.loadtxt(Field, usecols=(0,), skiprows= 0, unpack =True )
	for i in range (0, Nz):
		Frame[i] = E[i*Nx: i*Nx + Nx]
	
	Full[z] = Frame
	print(max(E))
	print(min(E))

norm = mpl.colors.Normalize(vmin=-1.0, vmax=1.0)
pc_kwargs = {'rasterized': True, 'cmap': 'jet'}
im = ax0.pcolormesh(X, Y, Full[0, :-1, :-1], norm=norm, **pc_kwargs)
cbar = fig.colorbar(im, ax=ax0)

def animate(i):	
	im.set_array(Full[i, :-1, :-1].flatten())

	# ax0.set_aspect(2)

	
	
	# ax0.axhline(y =50, color = 'silver', linewidth=1)

	# ax0.axhline(y =150, color = 'fuchsia', linewidth=1)
	# ax0.axhline(y =350, color = 'fuchsia', linewidth=1)

	# ax0.axhline(y =225, color = 'silver', linewidth=1)
	# ax0.axhline(y =250, color = 'silver', linewidth=1)

	# ax0.axhline(y =160, color = 'yellow', linewidth=1)


cbar.set_label(label = r"$\rm E \ \ Field \ (V m^{-1})$", size = '20')

#ax0.set_title(r"$ \rm  %.2f \ (\mu m)$" % fontsize = '25')
ax0.set_xlabel(r"$ \rm x \ $", fontsize = '20')
ax0.set_ylabel(r"$ \rm y \ $", fontsize = '20')

plt.setp(ax0.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)

# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)


#fig1 = plt.subplots(figsize = (10, 2),constrained_layout=True)
anim = FuncAnimation(fig, animate, interval = 50, frames = T - 1)
#plt.show()
anim.save("RepoMovie/RickerAgHoles3umCPML2.mp4")
	#plt.savefig("hBN_Ag_Si%dHz.pdf" %z)
	#plt.savefig("../EdgeImages/DomainWall/EdgeEz%d.png" %z)

