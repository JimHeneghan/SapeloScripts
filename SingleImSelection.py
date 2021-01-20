#!/usr/local/apps/anaconda/3-2.2.0/bin/python
import time 
import numpy as np
import scipy as sp
import itertools as it
import math
import collections as cl
import matplotlib.pyplot as plt
import matplotlib as mpl

# fig, ((ax00, ax01, ax02), (ax10, ax11, ax12), (ax20, ax21, ax22), (ax30, ax31, ax32)) = plt.subplots(4,3, figsize = (16,20))#, constrained_layout=True)
freq = ['6.1', '6.3', '6.5', '6.7', '6.8', '6.9', '7.0', '7.1', '7.2', '7.3', '7.5', '7.7', '8.0', '9.0', '9.5', '10.0', '10.5']
selectionBox = [4, 9, 11]

#______________________________________________________________________#
############################## Ex Field ################################
#______________________________________________________________________#
# fig, axs = plt.subplots(2, 5, figsize = (18, 9),constrained_layout=True)
fig, axs = plt.subplots(3, figsize = (6,25), constrained_layout=True)

Frame = []
Nx = 115
Ny = 199
z = 0
dt = 5.945e-18
T = 2e6
#print(z)
for ax0 in axs.flat:
	sbox = selectionBox[z]
	lam = freq[sbox]
	Field = "ExHeatMap%s.txt" %lam 
	# print(Field)



	Full = np.zeros((Ny, Nx), dtype = np.double)

	Y = np.linspace(-1.99, 1.99, Ny)
	X = np.linspace(-1.15, 1.15, Nx)

	E = np.loadtxt(Field, usecols=(0,), skiprows= 1, unpack =True )
	for i in range (0, Ny):
	 	Full[i] = E[i*Nx: i*Nx + Nx]*(dt/T)

	print(max(E)*(dt/T))
	norm = mpl.colors.Normalize(vmin=-1e-21, vmax=1e-21)
	pc_kwargs = {'rasterized': True, 'cmap': 'jet'}

	im = ax0.pcolormesh(X, Y, Full, norm=norm, **pc_kwargs)

	ax0.set_title(r"$\rm E_{x} : \ \lambda = %s \ \mu m $" %lam, fontsize = '35')
	ax0.set_aspect('equal')

	ax0.set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '35')
	ax0.set_ylabel(r"$ \rm y \ (\mu m)$", fontsize = '35')
	plt.setp(ax0.spines.values(), linewidth=2)
	ax0.tick_params(left = False, bottom = False, labelsize = "20")
	z = z + 1




	# plt.show()
cbar = fig.colorbar(im, ax=axs, orientation = 'horizontal')
cbar.set_label(label = r"$\rm Ex \ \ Field \ (V m^{-1})$", size = '35')
cbar.ax.tick_params(labelsize = 15)
plt.savefig("FullDev2_46um1ExHMs.pdf")
plt.savefig("FullDev2_46um1ExHMs.png")

print("\n \n")


#______________________________________________________________________#
############################## Ey Field ################################
#______________________________________________________________________#

fig, axs = plt.subplots(3, figsize = (6,25), constrained_layout=True)

Frame = []
Nx = 115
Ny = 199
z = 0
dt = 5.945e-18
T = 2e6
#print(z)
for ax0 in axs.flat:
	sbox = selectionBox[z]
	lam = freq[sbox]
	Field = "EyHeatMap%s.txt" %lam 

	Full = np.zeros((Ny, Nx), dtype = np.double)

	Y = np.linspace(-1.99, 1.99, Ny)
	X = np.linspace(-1.15, 1.15, Nx)

	E = np.loadtxt(Field, usecols=(0,), skiprows= 1, unpack =True )
	for i in range (0, Ny):
	 	Full[i] = E[i*Nx: i*Nx + Nx]*(dt/T)

	print(max(E)*(dt/T))
	norm = mpl.colors.Normalize(vmin=-1e-21, vmax=1e-21)
	pc_kwargs = {'rasterized': True, 'cmap': 'jet'}

	im = ax0.pcolormesh(X, Y, Full, norm=norm, **pc_kwargs)

	ax0.set_title(r"$\rm E_{y} : \ \lambda = %s \ \mu m $" %lam, fontsize = '35')
	ax0.set_aspect('equal')

	ax0.set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '35')
	ax0.set_ylabel(r"$ \rm y \ (\mu m)$", fontsize = '35')
	plt.setp(ax0.spines.values(), linewidth=2)
	ax0.tick_params(left = False, bottom = False, labelsize = "20")
	z = z + 1




	# plt.show()
cbar = fig.colorbar(im, ax=axs, orientation = 'horizontal')
cbar.set_label(label = r"$\rm Ey \ \ Field \ (V m^{-1})$", size = '35')
cbar.ax.tick_params(labelsize = 15)
plt.savefig("FullDev2_46umEyHMs.pdf")
plt.savefig("FullDev2_46umEyHMs.png")

print("\n \n")

# #______________________________________________________________________#
# ############################## Ez Selec ################################
# #______________________________________________________________________#

fig, axs = plt.subplots(3, figsize = (6,25), constrained_layout=True)

Frame = []
Nx = 115
Ny = 199
z = 0
dt = 5.945e-18
T = 2e6
#print(z)
for ax0 in axs.flat:
	sbox = selectionBox[z]
	lam = freq[sbox]
	Field = "EzHeatMap%s.txt" %lam 
	# print(Field)



	Full = np.zeros((Ny, Nx), dtype = np.double)

	Y = np.linspace(-1.99, 1.99, Ny)
	X = np.linspace(-1.15, 1.15, Nx)

	E = np.loadtxt(Field, usecols=(0,), skiprows= 1, unpack =True )
	for i in range (0, Ny):
	 	Full[i] = E[i*Nx: i*Nx + Nx]*(dt/T)

	norm = mpl.colors.Normalize(vmin=-1e-21, vmax=1e-21)
	pc_kwargs = {'rasterized': True, 'cmap': 'jet'}

	im = ax0.pcolormesh(X, Y, Full, norm=norm, **pc_kwargs)

	ax0.set_title(r"$\rm E_{z} : \ \lambda = %s \ \mu m $" %lam, fontsize = '35')
	ax0.set_aspect('equal')

	ax0.set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '35')
	ax0.set_ylabel(r"$ \rm y \ (\mu m)$", fontsize = '35')
	plt.setp(ax0.spines.values(), linewidth=2)
	ax0.tick_params(left = False, bottom = False, labelsize = "20")
	z = z + 1




	# plt.show()
cbar = fig.colorbar(im, ax=axs, orientation = 'horizontal')
cbar.set_label(label = r"$\rm Ez \ \ Field \ (V m^{-1})$", size = '35')
cbar.ax.tick_params(labelsize = 15)
plt.savefig("FullDev2_46umEzHMs.pdf")
plt.savefig("FullDev2_46umEzHMs.png")

print("\n \n")




# #______________________________________________________________________#
# ############################## |E| Field ###############################
# #______________________________________________________________________#

fig, axs = plt.subplots(3, figsize = (6,25), constrained_layout=True)

Frame = []
Nx = 115
Ny = 199
z = 0
dt = 5.945e-18
T = 2e6
#print(z)
for ax0 in axs.flat:
	sbox = selectionBox[z]
	lam = freq[sbox]
	Fieldx = "ExHeatMap%s.txt" %lam 
	Fieldy = "EyHeatMap%s.txt" %lam 
	Fieldz = "EzHeatMap%s.txt" %lam 

	Full = np.zeros((Ny, Nx), dtype = np.double)
	ExAb = np.zeros((Ny, Nx), dtype = np.double)
	EyAb = np.zeros((Ny, Nx), dtype = np.double)
	EzAb = np.zeros((Ny, Nx), dtype = np.double)

	Y = np.linspace(-1.99, 1.99, Ny)
	X = np.linspace(-1.15, 1.15, Nx)

	Ex = np.loadtxt(Fieldx, usecols=(0,), skiprows= 1, unpack =True )
	Ey = np.loadtxt(Fieldy, usecols=(0,), skiprows= 1, unpack =True )
	Ez = np.loadtxt(Fieldz, usecols=(0,), skiprows= 1, unpack =True )

	Ex = abs(Ex)*abs(Ex)*(dt/T)*(dt/T)
	Ey = abs(Ey)*abs(Ey)*(dt/T)*(dt/T)
	Ez = abs(Ez)*abs(Ez)*(dt/T)*(dt/T)

	for i in range (0, Ny):
		Full[i] = Ex[i*Nx: i*Nx + Nx] + Ey[i*Nx: i*Nx + Nx]+ Ez[i*Nx: i*Nx + Nx]
		Full[i] = np.sqrt(Full[i])
	
	norm = mpl.colors.Normalize(vmin=0, vmax=2e-21) 
	pc_kwargs = {'rasterized': True, 'cmap': 'jet'}

	im = ax0.pcolormesh(X, Y, Full, norm=norm, **pc_kwargs)

	ax0.set_title(r"$\rm |E| : \ \lambda = %s \ \mu m $" %lam, fontsize = '25')
	ax0.set_aspect('equal')

	ax0.set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '35')
	ax0.set_ylabel(r"$ \rm y \ (\mu m)$", fontsize = '35')
	plt.setp(ax0.spines.values(), linewidth=2)
	ax0.tick_params(left = False, bottom = False, labelsize = "20")
	z = z + 1


cbar = fig.colorbar(im, ax=axs, orientation = 'horizontal')
cbar.set_label(label = r"$\rm |E| \ \ Field \ (V m^{-1})$", size = '35')
cbar.ax.tick_params(labelsize = 15)
plt.savefig("FullDev2_46um2MagEHMs.pdf")
plt.savefig("FullDev2_46um2MagEHMs.png")

print("\n \n")
