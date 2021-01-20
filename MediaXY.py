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
fig, ax0 = plt.subplots(figsize = (4, 4), constrained_layout=True)
Frame = []
Nx = 23
Ny = 39
#print(z)
Field = "XY.dat" 
Full = np.zeros((Ny, Nx), dtype = np.double)

X = np.linspace(0,  2.3, Nx)
Y = np.linspace(0,  4, Ny)

E = np.loadtxt(Field, usecols=(0,), skiprows= 0, unpack =True )
print(max(E))
for i in range (0, Ny):
	for j in range (0, Nx):
 		Full[i][j] = E[i*Nx + j]

print(max(E))

cMap = c.ListedColormap(['tan', 'violet', 'black', 'silver', 'lightcoral'])
norm = mpl.colors.Normalize(vmin=0, vmax=4)
pc_kwargs = {'rasterized': True}

ax0.set_aspect(4/4.6)

im = ax0.pcolormesh(X, Y, Full, norm=norm, cmap = cMap, **pc_kwargs)
# ax0.axhline(y =Y[1260], color = 'fuchsia')
# ax0.axhline(y =Y[1268], color = 'fuchsia')
#cbar = fig.colorbar(im, ax=ax0)
#cbar.set_label(label = r"$\rm E \ \ Field \ (V m^{-1})$", size = '20')

#ax0.set_title(r"$ \rm  %.2f \ (\mu m)$" % fontsize = '25')
ax0.set_xlabel(r"$ \rm x \ $", fontsize = '20')
ax0.set_ylabel(r"$ \rm y \ $", fontsize = '20')

ax0.grid(True, which='minor', axis='both', linestyle='-', color='k')

ax0.set_xticks(X, minor=True)
ax0.set_yticks(Y, minor=True)

ax0.set_xlim(X[0], X[-1])
ax0.set_ylim(Y[0], Y[-1])

plt.savefig("XYAgPatternEvenStruct_min2.png")
plt.savefig("XYAgPatternEvenStruct_min2.pdf")

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

