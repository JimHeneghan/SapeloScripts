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
############################## |E| Field ###############################
#______________________________________________________________________#
for k in range(0,28):
	fig, axs = plt.subplots(5, 2, figsize = (12, 15),constrained_layout=True)
	Frame = []
	Nx = 111
	Ny = 1100
	z = 0
	dt = 5.945e-18
	T = 2e6
	base = 6.45 + k*0.1
	#print(z)
	for ax0 in axs.flat:
		lam = base + 0.01*z
		Fieldx = "Raw/ExHeatMap%.2f.txt" %lam 
		# Fieldy = "Raw/EyHeatMap%.2f.txt" %lam 
		# Fieldz = "Raw/EzHeatMap%.2f.txt" %lam 

		Full = np.zeros((400, Nx), dtype = np.double)
		ExAb = np.zeros((Ny, Nx), dtype = np.double)
		# EyAb = np.zeros((Ny, Nx), dtype = np.double)
		# EzAb = np.zeros((Ny, Nx), dtype = np.double)

		Y = np.linspace(0,  0.8, 400)
		X = np.linspace(0,  2.22, Nx)

		Ex = np.loadtxt(Fieldx, usecols=(0,), skiprows= 1, unpack =True )
		# Ey = np.loadtxt(Fieldy, usecols=(0,), skiprows= 1, unpack =True )
		# Ez = np.loadtxt(Fieldz, usecols=(0,), skiprows= 1, unpack =True )
		# Ex = Ex*Ex*(dt/T)*(dt/T)
		# Ey = Ey*Ey*(dt/T)*(dt/T)
		# Ez = Ez*Ez*(dt/T)*(dt/T)

		Ex = (Ex)*(dt/T)
		print(max(Ex))
		print(min(Ex))

		# Ey = abs(Ey)*abs(Ey)*(dt/T)*(dt/T)
		# Ez = abs(Ez)*abs(Ez)*(dt/T)*(dt/T)

		for i in range (350, 750):
			Full[i-350] = Ex[i*Nx: i*Nx + Nx]# + Ey[i*Nx: i*Nx + Nx]+ Ez[i*Nx: i*Nx + Nx]
			# Full[i-350] = np.sqrt(Full[i-350])
			# print(min(Full[i-350]))

		# print(max(np.sqrt(Ex)))
		
		norm = mpl.colors.Normalize(vmin=-3e-21, vmax=7e-21) 
		pc_kwargs = {'rasterized': True, 'cmap': 'jet'}

		im = ax0.pcolormesh(X, Y, Full, norm=norm, **pc_kwargs)
		ax0.set_aspect(1.0/0.8)
		ax0.set_title(r"$\rm \lambda_{E_{x}} = %.2f \ \mu m $" %lam, fontsize = '25')

		ax0.set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '20')
		ax0.set_ylabel(r"$ \rm y \ (\mu m)$", fontsize = '20')
		plt.setp(ax0.spines.values(), linewidth=2)
		plt.tick_params(left = False, bottom = False)
		z = z + 1

	cbar = fig.colorbar(im, ax=axs)
	cbar.set_label(label = r"$\rm |E| \ \ Field \ (V m^{-1})$", size = '20')

	plt.savefig("ExMaps/PDF/Ex2.22XSecHM%.2f_%.2f_3_7Mag.pdf" %(base, lam))
	plt.savefig("ExMaps/PNG/Ex2.22XSecHM%.2f_%.2f_3_7Mag.png" %(base, lam))
#abs(Full*(dt/T))


#______________________________________________________________________#
############################## |E| Field ###############################
#______________________________________________________________________#

# fig, axs = plt.subplots(5, 2, figsize = (12, 15),constrained_layout=True)
# Frame = []
# Nx = 115
# Ny = 1100
# z = 0
# dt = 5.945e-18
# T = 2e6
# #print(z)
# for ax0 in axs.flat:
# 	lam = 7.0 + 0.1*z
# 	Fieldx = "ExHeatMap%s.txt" %lam 
# 	Fieldy = "EyHeatMap%s.txt" %lam 
# 	Fieldz = "EzHeatMap%s.txt" %lam 

# 	Full = np.zeros((400, Nx), dtype = np.double)
# 	ExAb = np.zeros((Ny, Nx), dtype = np.double)
# 	EyAb = np.zeros((Ny, Nx), dtype = np.double)
# 	EzAb = np.zeros((Ny, Nx), dtype = np.double)

# 	Y = np.linspace(0,  0.8, 400)
# 	X = np.linspace(0,  2.22, Nx)

# 	Ex = np.loadtxt(Fieldx, usecols=(0,), skiprows= 1, unpack =True )
# 	Ey = np.loadtxt(Fieldy, usecols=(0,), skiprows= 1, unpack =True )
# 	Ez = np.loadtxt(Fieldz, usecols=(0,), skiprows= 1, unpack =True )

# 	Ex = Ex*Ex*(dt/T)*(dt/T)
# 	Ey = Ey*Ey*(dt/T)*(dt/T)
# 	Ez = Ez*Ez*(dt/T)*(dt/T)

# 	for i in range (350, 750):
# 		Full[i-350] = Ex[i*Nx: i*Nx + Nx] + Ey[i*Nx: i*Nx + Nx]+ Ez[i*Nx: i*Nx + Nx]
# 		Full[i-350] = np.sqrt(Full[i-350])
# 		# print(max(Full[i-350]))

# 	# print(max(np.sqrt(Ex)))
	
# 	norm = mpl.colors.Normalize(vmin=0, vmax=5e-21) 
# 	pc_kwargs = {'rasterized': True, 'cmap': 'jet'}

# 	im = ax0.pcolormesh(X, Y, Full, norm=norm, **pc_kwargs)
# 	ax0.set_aspect(1.0/0.8)

# 	ax0.set_title(r"$\rm \lambda_{|E|} = %s \ 0 \ \mu m $" %lam, fontsize = '25')

# 	ax0.set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '20')
# 	ax0.set_ylabel(r"$ \rm y \ (\mu m)$", fontsize = '20')
# 	plt.setp(ax0.spines.values(), linewidth=2)
# 	plt.tick_params(left = False, bottom = False)
# 	z = z + 1

# cbar = fig.colorbar(im, ax=axs)
# cbar.set_label(label = r"$\rm |E| \ \ Field \ (V m^{-1})$", size = '20')
# # plt.savefig("EabsTest2.png")

# plt.savefig("ZoomFullDev2_22um_Xsec_MagE_BarSource7_8.pdf")
# plt.savefig("ZoomFullDev2_22um_Xsec_MagE_BarSource7_8.png")


# #______________________________________________________________________#
# ############################## |E| Field ###############################
# #______________________________________________________________________#

# fig, axs = plt.subplots(5, 2, figsize = (12, 15),constrained_layout=True)
# Frame = []
# Nx = 115
# Ny = 1100
# z = 0
# dt = 5.945e-18
# T = 2e6
# #print(z)
# for ax0 in axs.flat:
# 	lam = 8.0 + 0.1*z
# 	Fieldx = "ExHeatMap%s.txt" %lam 
# 	Fieldy = "EyHeatMap%s.txt" %lam 
# 	Fieldz = "EzHeatMap%s.txt" %lam 

# 	Full = np.zeros((400, Nx), dtype = np.double)
# 	ExAb = np.zeros((Ny, Nx), dtype = np.double)
# 	EyAb = np.zeros((Ny, Nx), dtype = np.double)
# 	EzAb = np.zeros((Ny, Nx), dtype = np.double)

# 	Y = np.linspace(0,  0.8, 400)
# 	X = np.linspace(0,  2.22, Nx)

# 	Ex = np.loadtxt(Fieldx, usecols=(0,), skiprows= 1, unpack =True )
# 	Ey = np.loadtxt(Fieldy, usecols=(0,), skiprows= 1, unpack =True )
# 	Ez = np.loadtxt(Fieldz, usecols=(0,), skiprows= 1, unpack =True )

# 	Ex = Ex*Ex*(dt/T)*(dt/T)
# 	Ey = Ey*Ey*(dt/T)*(dt/T)
# 	Ez = Ez*Ez*(dt/T)*(dt/T)

# 	for i in range (350, 750):
# 		Full[i-350] = Ex[i*Nx: i*Nx + Nx] + Ey[i*Nx: i*Nx + Nx]+ Ez[i*Nx: i*Nx + Nx]
# 		Full[i-350] = np.sqrt(Full[i-350])
# 		# print(max(Full[i-350]))

# 	# print(max(np.sqrt(Ex)))
	
# 	norm = mpl.colors.Normalize(vmin=0, vmax=5e-21) 
# 	pc_kwargs = {'rasterized': True, 'cmap': 'jet'}

# 	im = ax0.pcolormesh(X, Y, Full, norm=norm, **pc_kwargs)

# 	ax0.set_title(r"$\rm \lambda_{|E|} = %s \ 0 \ \mu m $" %lam, fontsize = '25')
# 	ax0.set_aspect(1.0/0.8)

# 	ax0.set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '20')
# 	ax0.set_ylabel(r"$ \rm y \ (\mu m)$", fontsize = '20')
# 	plt.setp(ax0.spines.values(), linewidth=2)
# 	plt.tick_params(left = False, bottom = False)
# 	z = z + 1

# cbar = fig.colorbar(im, ax=axs)
# cbar.set_label(label = r"$\rm |E| \ \ Field \ (V m^{-1})$", size = '20')
# # plt.savefig("EabsTest2.png")

# plt.savefig("ZoomFullDev2_22um_Xsec_MagE_BarSource8_9.pdf")
# plt.savefig("ZoomFullDev2_22um_Xsec_MagE_BarSource8_9.png")