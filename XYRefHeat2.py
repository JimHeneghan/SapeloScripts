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
Frame = []
Nx = 45
Ny = 79
NFREQs = 200
#print(z)

fig, ax0 = plt.subplots(figsize = (3, 3),constrained_layout=True)

Field = "ExRef.txt"

# Full = np.zeros((Nx, Ny), dtype = np.double)

Y = np.linspace(0,  Nx, Nx)
X = np.linspace(0,  Ny, Ny)

E = np.loadtxt(Field, usecols=(0,), skiprows= 1, unpack =True )
# for i in range (0, Nx):
#  	Full[i] = E[i*Ny: i*Ny + Ny]

Full  = np.reshape(E, (NFREQs, Nx, Ny), order='C')
print(max(E))
print("\n")

print(min(E))

norm = mpl.colors.Normalize(vmin=-3e4, vmax=3e4)
pc_kwargs = {'rasterized': True, 'cmap': 'jet'}

ax0.set_aspect(4/2.3)

im = ax0.pcolormesh(X, Y, Full[1], norm=norm, **pc_kwargs)
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

# plt.show()
plt.savefig("AgHeatRefSpec2.pdf")
plt.savefig("AgHeatRefSpec2.png")
