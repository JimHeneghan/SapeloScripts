import time 
import numpy as np
import scipy as sp
import itertools as it
import math
import collections as cl
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
#from mpl_toolkits.axes_grid1 import make_axes_locatable
freq = ['6.3', '6.4', '6.5', '6.75', '6.9', '7.0', '7.1', '7.2', '7.5', '8.0']
# many = np.zeros((199, 115), dtype = np.double)
j = 0
z = 4
pc_kwargs = {'rasterized': True, 'cmap': 'jet'}
fig, ax = plt.subplots(1, figsize = (4, 4), constrained_layout=True)#, wspace = 0.1)
# fig.suptitle(r"$\rm hBN \ Layer$", fontsize = '25')
# for ax in axs.flat:
Field = 'RefSpec.txt'
print(z)
# if (z == 4)or(z == 5)or(z == 10)or(z == 11):
# 	Full = np.zeros((401, 253), dtype = np.double)
# 	X = np.linspace(0,  2.53, 253)
# 	Y = np.linspace(0,  4.01, 401)
# 	ax.set_aspect(401/253)
# 	for i in range (0, 400):
# 		E = np.loadtxt(Field, usecols=(i,), skiprows= 697, unpack =True )
# 		Full[i] = E 
# else:
Full = np.zeros((199, 114), dtype = np.double)

X = np.linspace(0,  2.3, 114)
Y = np.linspace(0,  4, 199)

E = np.loadtxt(Field, usecols=(1,), skiprows= 1, unpack =True )
print(max(E))
for i in range (0, 199):
	for j in range (0, 114):
 		Full[i][j] = abs(E[i*114 + j])
print(freq[z])
print(max(E))
ax.set_aspect(1.1)
norm = mpl.colors.Normalize(vmin=0, vmax=0.347)
im = ax.pcolormesh(X, Y, Full, norm = norm, **pc_kwargs)

ax.set_title(r"$\rm \lambda_{Ex} = %s \ \mu m $" %freq[z], fontsize = '25')

# if (z==0):
# 	ax.set_title(r"$\rm \lambda_{Ex} = 8.57 \ \mu m $", fontsize = '17')
# elif (z==1):
# 	ax.set_title(r"$\rm \lambda_{Ex} = 7.69 \ \mu m $", fontsize = '17')
# elif (z==2):
# 	ax.set_title(r"$\rm \lambda_{Ex} = 6.82 \ \mu m $", fontsize = '17')
# elif (z==3):
# 	ax.set_title(r"$\rm \lambda_{Ex} = 6.38 \ \mu m $", fontsize = '17')
	

ax.set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '20')
ax.set_ylabel(r"$ \rm y \ (\mu m)$", fontsize = '20')
plt.setp(ax.spines.values(), linewidth=2)
ax.tick_params(left = False, bottom = False)

z = z + 1


	

cbar = fig.colorbar(im, ax=ax)
cbar.set_label(label = r"$\rm \|E\|  \ (V m^{-1})$", size = '20')

# plt.tight_layout()
plt.savefig("HeatMapMe.pdf")
plt.savefig("HeatMapMe.png")
# plt.show()
