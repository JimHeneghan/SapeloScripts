#!/usr/local/apps/anaconda/3-2.2.0/bin/python
import time 
import numpy as np
import scipy as sp
import itertools as it
import math
import collections as cl
import matplotlib.pyplot as plt
import matplotlib as mpl


#______________________________________________________________________#
############################## Ex Field ################################
#______________________________________________________________________#
# fig, axs = plt.subplots(2, 5, figsize = (18, 9),constrained_layout=True)
# Frame = []
# Nx = 115
# Ny = 199
# z = 0
dt = 5.945e-18
T = 2e6

#______________________________________________________________________#
############################## Ex Again ################################
#______________________________________________________________________#

fig, axs = plt.subplots(2, 5, figsize = (18, 9),constrained_layout=True)
Frame = []
Nx = 115
Ny = 199
z = 0
#print(z)
for ax0 in axs.flat:
	lam = 6.0 + 0.1*z
	Field = "Raw/ExHeatMap%s0.txt" %lam 
	
	Full = np.zeros((Ny, Nx), dtype = np.double)

	Y = np.linspace(0,  Ny, Ny)
	X = np.linspace(0,  Nx, Nx)

	E = np.loadtxt(Field, usecols=(0,), skiprows= 1, unpack =True )
	for i in range (0, Ny):
	 	Full[i] = E[i*Nx: i*Nx + Nx]*(dt/T)

	print(max(E)*(dt/T))
	norm = mpl.colors.Normalize(vmin=-5e-21, vmax=5e-21)
	pc_kwargs = {'rasterized': True, 'cmap': 'jet'}

	im = ax0.pcolormesh(X, Y, Full, norm=norm, **pc_kwargs)

	ax0.set_title(r"$\rm \lambda_{Ex} = %s \ \mu m $" %lam, fontsize = '25')

	ax0.set_xlabel(r"$ \rm x \ $", fontsize = '20')
	ax0.set_ylabel(r"$ \rm y \ $", fontsize = '20')
	plt.setp(ax0.spines.values(), linewidth=2)
	plt.tick_params(left = False, bottom = False)
	z = z + 1

cbar = fig.colorbar(im, ax=axs)
cbar.set_label(label = r"$\rm Ex \ \ Field \ (V m^{-1})$", size = '20')
plt.savefig("FullDev2_46umEx2_BarSource.pdf")
plt.savefig("FullDev2_46umEx2_BarSource.png")

print("\n \n")

#______________________________________________________________________#
############################## Ex Again 7-8 um #########################
#______________________________________________________________________#

fig, axs = plt.subplots(2, 5, figsize = (18, 9),constrained_layout=True)
Frame = []
Nx = 115
Ny = 199
z = 0
#print(z)
for ax0 in axs.flat:
	lam = 7.0 + 0.1*z
	Field = "Raw/ExHeatMap%s0.txt" %lam 
	
	Full = np.zeros((Ny, Nx), dtype = np.double)

	Y = np.linspace(0,  Ny, Ny)
	X = np.linspace(0,  Nx, Nx)

	E = np.loadtxt(Field, usecols=(0,), skiprows= 1, unpack =True )
	for i in range (0, Ny):
	 	Full[i] = E[i*Nx: i*Nx + Nx]*(dt/T)

	print(max(E)*(dt/T))
	norm = mpl.colors.Normalize(vmin=-5e-21, vmax=5e-21)
	pc_kwargs = {'rasterized': True, 'cmap': 'jet'}

	im = ax0.pcolormesh(X, Y, Full, norm=norm, **pc_kwargs)

	ax0.set_title(r"$\rm \lambda_{Ex} = %s \ \mu m $" %lam, fontsize = '25')

	ax0.set_xlabel(r"$ \rm x \ $", fontsize = '20')
	ax0.set_ylabel(r"$ \rm y \ $", fontsize = '20')
	plt.setp(ax0.spines.values(), linewidth=2)
	plt.tick_params(left = False, bottom = False)
	z = z + 1

cbar = fig.colorbar(im, ax=axs)
cbar.set_label(label = r"$\rm Ex \ \ Field \ (V m^{-1})$", size = '20')
plt.savefig("FullDev2_46umEx2_BarSource7_8.pdf")
plt.savefig("FullDev2_46umEx2_BarSource7_8.png")

print("\n \n")
#______________________________________________________________________#
############################## Ex Again 8-9 um #########################
#______________________________________________________________________#

fig, axs = plt.subplots(5, 2, figsize = (18, 9),constrained_layout=True)
Frame = []
Nx = 115
Ny = 199
z = 0
#print(z)
for ax0 in axs.flat:
	lam = 8.0 + 0.1*z
	Field = "Raw/ExHeatMap%s0.txt" %lam 
	
	Full = np.zeros((Ny, Nx), dtype = np.double)

	Y = np.linspace(0,  Ny, Ny)
	X = np.linspace(0,  Nx, Nx)

	E = np.loadtxt(Field, usecols=(0,), skiprows= 1, unpack =True )
	for i in range (0, Ny):
	 	Full[i] = E[i*Nx: i*Nx + Nx]*(dt/T)

	print(max(E)*(dt/T))
	norm = mpl.colors.Normalize(vmin=-5e-21, vmax=5e-21)
	pc_kwargs = {'rasterized': True, 'cmap': 'jet'}

	im = ax0.pcolormesh(X, Y, Full, norm=norm, **pc_kwargs)

	ax0.set_title(r"$\rm \lambda_{Ex} = %s \ \mu m $" %lam, fontsize = '25')

	ax0.set_xlabel(r"$ \rm x \ $", fontsize = '20')
	ax0.set_ylabel(r"$ \rm y \ $", fontsize = '20')
	plt.setp(ax0.spines.values(), linewidth=2)
	plt.tick_params(left = False, bottom = False)
	z = z + 1

cbar = fig.colorbar(im, ax=axs)
cbar.set_label(label = r"$\rm Ex \ \ Field \ (V m^{-1})$", size = '20')
plt.savefig("FullDev2_46umEx2_BarSource8_9.pdf")
plt.savefig("FullDev2_46umEx2_BarSource8_9.png")

print("\n \n")
