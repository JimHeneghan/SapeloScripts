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
from matplotlib import colors as c
#from mpl_toolkits.axes_grid1 import make_axes_locatable

#many = np.zeros((10, 401, 231), dtype = np.double)
fig, ax0 = plt.subplots(figsize = (3, 3), constrained_layout=True)
Frame = []
Ny = 39
Nz = 1020
#print(z)
Field = "Media.dat" 
Full = np.zeros((Nz, Ny), dtype = np.double)

X = np.linspace(0,  3.95, Ny)
Y = np.linspace(0,  5.1, Nz)

E = np.loadtxt(Field, usecols=(1,), skiprows= 0, unpack =True )
print(max(E))
for i in range (0, Nz):
	for j in range (0, Ny):
 		Full[i][j] = E[i*Ny + j]

print(max(E))
cMap = c.ListedColormap(['tan', 'violet', 'black', 'silver', 'lightcoral'])
norm = mpl.colors.Normalize(vmin=0, vmax=4)
pc_kwargs = {'rasterized': True}

ax0.set_aspect(4/5)

im = ax0.pcolormesh(X, Y, Full, norm=norm, cmap = cMap, **pc_kwargs)
# ax0.axhline(y =Y[1260], color = 'fuchsia')
# ax0.axhline(y =Y[1268], color = 'fuchsia')
#cbar = fig.colorbar(im, ax=ax0)
#cbar.set_label(label = r"$\rm E \ \ Field \ (V m^{-1})$", size = '20')

#ax0.set_title(r"$ \rm  %.2f \ (\mu m)$" % fontsize = '25')
ax0.set_xlabel(r"$ \rm x \ $", fontsize = '20')
ax0.set_ylabel(r"$ \rm y \ $", fontsize = '20')

plt.setp(ax0.spines.values(), linewidth=0)
plt.tick_params(left = False, bottom = False)

plt.savefig("CPMLYZ2MDrD.png")
plt.savefig("CPMLYZ2MDrD.pdf")

# Frame.append([im])
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
# namer = input("name the movie file: ")

# #fig1 = plt.subplots(figsize = (10, 2),constrained_layout=True)
# anim = animation.ArtistAnimation(fig, Frame, interval = 20)
# #plt.show()
# anim.save("../" + namer + ".mp4", writer = writer )
	#plt.savefig("hBN_Ag_Si%dHz.pdf" %z)
	#plt.savefig("../EdgeImages/DomainWall/EdgeEz%d.png" %z)

